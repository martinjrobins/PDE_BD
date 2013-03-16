/* 
 * Pde.cpp
 *
 * Copyright 2012 Martin Robinson
 *
 * This file is part of PDE_BD.
 *
 * PDE_BD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PDE_BD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with PDE_BD.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Created on: Feb 9, 2013
 *      Author: mrobins
 */

#include "Pde.h"
#include <iostream>

#include "Log.h"
#include "Constants.h"
#include "MyMpi.h"
#include <vtkPointData.h>
#include <vtkQuad.h>
#include <set>


#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_RTI.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>

const Pde::ST Pde::omega = 1.0;

Pde::Pde(const ST dt, const ST dx,const int test_no,const int numThreads):dt(dt),dirac_width(dx) {
	typedef Teuchos::ScalarTraits<ST> STS;
	typedef STS::magnitudeType MT;
	typedef Teuchos::ScalarTraits<MT> STM;

	my_rank = Mpi::mpiSession->getRank();
	num_procs = Mpi::mpiSession->getNProc();

	// Get the default communicator and Kokkos Node instance
	comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
	node = Tpetra::DefaultPlatform::getDefaultPlatform ().getNode ();
	node->init(numThreads);

	std::string meshInput;
	if (test_no == 1) {
		meshInput = makeMeshInput(1.0/dx, 1.0/dx, 1.0/dx);
	} else if (test_no == 2) {
		//meshInput = makeMeshInputRadialTrisection(ro/dx,0.5*3.14*ro/dx,1.0/dx);
		//meshInput = makeMeshInputSphere((ro-ri)/dx, 0.5*3.14*0.5*(ro+ri)/dx);
		meshInput = makeMeshInputCylinder((ro-ri)/dx,0.5*3.14*0.5*(ro+ri)/dx,1.0/dx);
		std::cout << meshInput;
	} else if (test_no == 3) {
		meshInput = makeMeshInputFullDomain(2.0/dx, 1.0/dx, 1.0/dx);
	}

	setup_pamgen_mesh(meshInput);
	create_cubature_and_basis();
	build_maps_and_create_matrices();
	create_vtk_grid();
	fillMatricies();
	calculate_volumes_and_areas();
	compose_LHS_and_RHS();
}

void Pde::add_particle(const ST x, const ST y, const ST z) {
	typedef Teuchos::ScalarTraits<ST> STS;
	typedef STS::magnitudeType MT;
	typedef Teuchos::ScalarTraits<MT> STM;

	const int numNodes = node_coord.dimension(0);
	int closest_node = 0;
	double closest_distance = 10000.0;
	for (int i=0; i<numNodes; i++) {
		if (node_is_owned[i]) {
			const ST dx = node_coord(i,0) - x;
			const ST dy = node_coord(i,1) - y;
			const ST dz = node_coord(i,2) - z;
			if ((dx < closest_distance) && (dy < closest_distance) && (dz < closest_distance)) {
				const MT r = STM::squareroot(dx*dx + dy*dy + dz*dz);
				if (r < closest_distance) {
					closest_distance = r;
					closest_node = i;
//					}
				} // if node within particle radius
			} // if node within the square
		} // if node is owned by this process
	} // loop through all nodes
	//std::cout << "adding "<<1.0/interior_node_volumes->get1dView()[closest_node]<< " to node "<<closest_node<<std::endl;
	U->sumIntoLocalValue(closest_node, 1.0/interior_node_volumes->get1dView()[closest_node]);
}


void Pde::setup_pamgen_mesh(const std::string& meshInput){
	using namespace Intrepid;
	using Teuchos::Array;
	/**********************************************************************************/
	/***************************** GET CELL TOPOLOGY **********************************/
	/**********************************************************************************/

	LOG(2,"Getting cell topology");

		// Get cell topology for base hexahedron
	cellType = shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> > ());
	faceType = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> > ());

	// Get dimensions
	int numNodesPerElem = cellType.getNodeCount();
	int numFacesPerElem = cellType.getSideCount();
	int numEdgesPerElem = cellType.getEdgeCount();
	numNodesPerFace = 4;
	int spaceDim = cellType.getDimension();
	int dim = 3;

	// Build reference element face to node map

	refFaceToNode.resize(numFacesPerElem,numNodesPerFace);
	for (int i=0; i<numFacesPerElem; i++){
		for (int j = 0; j < numNodesPerFace; ++j) {
			refFaceToNode(i,j)=cellType.getNodeMap(spaceDim-1, i, j);
		}
	}

	/**********************************************************************************/
	/******************************* GENERATE MESH ************************************/
	/**********************************************************************************/

	LOG(2,"Generating mesh");

	int error = 0; // Number of errors in generating the mesh

	long long *  node_comm_proc_ids   = NULL;
	long long *  node_cmap_node_cnts  = NULL;
	long long *  node_cmap_ids        = NULL;
	long long ** comm_node_ids        = NULL;
	long long ** comm_node_proc_ids   = NULL;

	// Generate mesh with Pamgen
	long long maxInt = 9223372036854775807LL;
	Create_Pamgen_Mesh (meshInput.c_str (), dim, my_rank, num_procs, maxInt);

	std::string msg ("Poisson: ");

	// Get mesh size info
	char title[100];
	long long numDim;
	long long numNodes;
	long long numElems;
	long long numElemBlk;
	long long numNodeSets;
	long long numSideSets;
	int id = 0;

	im_ex_get_init_l (id, title, &numDim, &numNodes, &numElems, &numElemBlk,
			&numNodeSets, &numSideSets);

	ASSERT(numElems > 0,"The number of elements in the mesh is zero.");


	long long numElemsGlobal;
	long long numElemBlkGlobal;
	long long numNodeSetsGlobal;
	long long numSideSetsGlobal;

	im_ne_get_init_global_l (id, &numNodesGlobal, &numElemsGlobal,
			&numElemBlkGlobal, &numNodeSetsGlobal,
			&numSideSetsGlobal);


	LOG(2,"Global number of elements:                     "
			<< numElemsGlobal << std::endl
			<< "Global number of nodes (incl. boundary nodes): "
			<< numNodesGlobal);


	long long * block_ids = new long long [numElemBlk];
	error += im_ex_get_elem_blk_ids_l(id, block_ids);

	long long  *nodes_per_element   = new long long [numElemBlk];
	long long  *element_attributes  = new long long [numElemBlk];
	long long  *elements            = new long long [numElemBlk];
	char      **element_types       = new char * [numElemBlk];
	long long **elmt_node_linkage   = new long long * [numElemBlk];

	for (long long i = 0; i < numElemBlk; ++i) {
		element_types[i] = new char [MAX_STR_LENGTH + 1];
		error += im_ex_get_elem_block_l (id,
				block_ids[i],
				element_types[i],
				(long long*)&(elements[i]),
				(long long*)&(nodes_per_element[i]),
				(long long*)&(element_attributes[i]));
	}

	// connectivity
	for (long long b = 0; b < numElemBlk; ++b) {
		elmt_node_linkage[b] =  new long long [nodes_per_element[b]*elements[b]];
		error += im_ex_get_elem_conn_l (id,block_ids[b], elmt_node_linkage[b]);
	}

	// Get node-element connectivity
	int telct = 0;
	elem_to_node.resize(numElems,numNodesPerElem);
	for (long long b = 0; b < numElemBlk; b++) {
		for (long long el = 0; el < elements[b]; el++) {
			for (int j = 0; j < numNodesPerElem; ++j) {
				elem_to_node(telct,j) = elmt_node_linkage[b][el*numNodesPerElem + j]-1;
			}
			++telct;
		}
	}

	// Read node coordinates and place in field container
	node_coord.resize(numNodes,dim);
	ST * nodeCoordx = new ST [numNodes];
	ST * nodeCoordy = new ST [numNodes];
	ST * nodeCoordz = new ST [numNodes];
	im_ex_get_coord_l (id, nodeCoordx, nodeCoordy, nodeCoordz);
	for (int i=0; i<numNodes; i++) {
		node_coord(i,0)=nodeCoordx[i];
		node_coord(i,1)=nodeCoordy[i];
		node_coord(i,2)=nodeCoordz[i];
	}
	delete [] nodeCoordx;
	delete [] nodeCoordy;
	delete [] nodeCoordz;

	// parallel info
	long long num_internal_nodes;
	long long num_border_nodes;
	long long num_external_nodes;
	long long num_internal_elems;
	long long num_border_elems;
	long long num_node_comm_maps;
	long long num_elem_comm_maps;
	im_ne_get_loadbal_param_l( id,
			&num_internal_nodes,
			&num_border_nodes,
			&num_external_nodes,
			&num_internal_elems,
			&num_border_elems,
			&num_node_comm_maps,
			&num_elem_comm_maps,
			0/*unused*/ );

	if (num_node_comm_maps > 0) {
		node_comm_proc_ids   = new long long  [num_node_comm_maps];
		node_cmap_node_cnts  = new long long  [num_node_comm_maps];
		node_cmap_ids        = new long long  [num_node_comm_maps];
		comm_node_ids        = new long long* [num_node_comm_maps];
		comm_node_proc_ids   = new long long* [num_node_comm_maps];

		long long *  elem_cmap_ids        = new long long [num_elem_comm_maps];
		long long *  elem_cmap_elem_cnts  = new long long [num_elem_comm_maps];

		if (im_ne_get_cmap_params_l (id,
				node_cmap_ids,
				(long long*)node_cmap_node_cnts,
				elem_cmap_ids,
				(long long*)elem_cmap_elem_cnts,
				0/*not used proc_id*/ ) < 0) {
			++error;
		}

		for (long long j = 0; j < num_node_comm_maps; ++j) {
			comm_node_ids[j]       = new long long [node_cmap_node_cnts[j]];
			comm_node_proc_ids[j]  = new long long [node_cmap_node_cnts[j]];
			if (im_ne_get_node_cmap_l (id,
					node_cmap_ids[j],
					comm_node_ids[j],
					comm_node_proc_ids[j],
					0/*not used proc_id*/ ) < 0) {
				++error;
			}
			node_comm_proc_ids[j] = comm_node_proc_ids[j][0];
		}

		delete [] elem_cmap_ids;
		delete [] elem_cmap_elem_cnts;
	}

	//
	// Calculate global node ids
	//
	global_node_ids.resize(numNodes,0);

	node_is_owned = new bool [numNodes];
	calc_global_node_ids (global_node_ids.getRawPtr (),
			node_is_owned,
			numNodes,
			num_node_comm_maps,
			node_cmap_node_cnts,
			node_comm_proc_ids,
			comm_node_ids,
			my_rank);



	//
	// Mesh cleanup
	//
	delete [] block_ids;
	block_ids = NULL;
	delete [] nodes_per_element;
	nodes_per_element = NULL;
	delete [] element_attributes;
	element_attributes = NULL;
	for (long long b = 0; b < numElemBlk; ++b) {
		delete [] elmt_node_linkage[b];
		delete [] element_types[b];
	}
	delete [] element_types;
	element_types = NULL;
	delete [] elmt_node_linkage;
	elmt_node_linkage = NULL;
	if (num_node_comm_maps > 0) {
		delete [] node_comm_proc_ids;
		node_comm_proc_ids = NULL;
		delete [] node_cmap_node_cnts;
		node_cmap_node_cnts = NULL;
		delete [] node_cmap_ids;
		node_cmap_ids = NULL;
		for (long long i = 0; i < num_node_comm_maps; ++i) {
			delete [] comm_node_ids[i];
			delete [] comm_node_proc_ids[i];
		}
		delete [] comm_node_ids;
		comm_node_ids = NULL;
		delete [] comm_node_proc_ids;
		comm_node_proc_ids = NULL;
	}
	delete [] elements;
	elements = NULL;

	// Container indicating whether a node is on the boundary (1-yes 0-no)
	node_on_boundary.resize(numNodes);
	node_on_neumann.resize(numNodes);

	//faceOnBoundary.resize(numFaces);

	// Get boundary (side set) information
	long long * sideSetIds = new long long [numSideSets];
	long long * numSidesInSet = new long long [numSideSets];
	long long numDFinSet;
	int numBndyFaces=0;
	im_ex_get_side_set_ids_l(id,sideSetIds);

	int i_boundary_face = 0;
	for (int i = 0; i < numSideSets; ++i) {
		im_ex_get_side_set_param_l (id,sideSetIds[i], &numSidesInSet[i], &numDFinSet);
		if (numSidesInSet[i] > 0) {
			long long * sideSetElemList = new long long [numSidesInSet[i]];
			long long * sideSetSideList = new long long [numSidesInSet[i]];
			im_ex_get_side_set_l (id, sideSetIds[i], sideSetElemList, sideSetSideList);

			for (int j = 0; j < numSidesInSet[i]; ++j) {
				const int iface = sideSetSideList[j]-1;
				const int ielem = sideSetElemList[j]-1;
				bool on_x_equal_one = true;
				bool on_outer_boundary = true;
				for (int ifacenode = 0; ifacenode < numNodesPerFace; ++ifacenode) {
					const int sideNode = cellType.getNodeMap(2,iface,ifacenode);
					const int local_nodeid = elem_to_node(ielem,sideNode);
					on_x_equal_one &= node_coord(local_nodeid,0)==1.0;
					on_outer_boundary &= node_coord(local_nodeid,0)==0.0 ||
										 node_coord(local_nodeid,0)==2.0 ||
										 node_coord(local_nodeid,1)==0.0 ||
										 node_coord(local_nodeid,1)==1.0 ||
										 node_coord(local_nodeid,2)==0.0 ||
										 node_coord(local_nodeid,2)==1.0;
				}
				if (1) {
					i_boundary_face++;
				}
			}
		}
	}

	boundary_face_to_elem.resize(i_boundary_face);
	boundary_face_to_ordinal.resize(i_boundary_face);

	i_boundary_face = 0;
	for (int i = 0; i < numSideSets; ++i) {
		if (numSidesInSet[i] > 0){
			long long * sideSetElemList = new long long [numSidesInSet[i]];
			long long * sideSetSideList = new long long [numSidesInSet[i]];
			im_ex_get_side_set_l (id, sideSetIds[i], sideSetElemList, sideSetSideList);

			for (int j = 0; j < numSidesInSet[i]; ++j) {
				const int iface = sideSetSideList[j]-1;
				const int ielem = sideSetElemList[j]-1;

				bool face_on_x_equal_one = true;
				bool face_on_neumann = true;
				bool face_on_outer_boundary = true;
				for (int ifacenode = 0; ifacenode < numNodesPerFace; ++ifacenode) {
					const int sideNode = cellType.getNodeMap(2,iface,ifacenode);
					const int local_nodeid = elem_to_node(ielem,sideNode);
					const double x = node_coord(local_nodeid,0);
					const double y = node_coord(local_nodeid,1);
					const double z = node_coord(local_nodeid,2);
					bool on_x_equal_one = x==1.0;
					bool on_neumann = y==0.0 || y==1.0 || z==0.0 || z==1.0 || x==0.0;
					bool on_outer_boundary = x==0.0 ||
							x==2.0 ||
							y==0.0 ||
							y==1.0 ||
							z==0.0 ||
							z==1.0;

					if (on_outer_boundary) {
						node_on_neumann(local_nodeid) = 1;
					} else {
						node_on_neumann(local_nodeid) = 0;
					}

					face_on_neumann &= on_neumann;
					face_on_x_equal_one &= on_x_equal_one;
					face_on_outer_boundary &= on_outer_boundary;
				}

				if (1) {
					boundary_face_to_elem(i_boundary_face) = ielem;
					boundary_face_to_ordinal(i_boundary_face) = iface;
					for (int ifacenode = 0; ifacenode < numNodesPerFace; ++ifacenode) {
						const int sideNode = cellType.getNodeMap(2,iface,ifacenode);
						const int local_nodeid = elem_to_node(ielem,sideNode);
						node_on_boundary(local_nodeid) = 1;
					}

					i_boundary_face++;
				}
			}
			delete [] sideSetElemList;
			delete [] sideSetSideList;
		}
	}
	delete [] sideSetIds;
	delete [] numSidesInSet;





	//
	// We're done with assembly, so we can delete the mesh.
	//

	Delete_Pamgen_Mesh ();
}

void Pde::create_cubature_and_basis() {
	using namespace Intrepid;
	/**********************************************************************************/
	/********************************* GET CUBATURE For 3D cells***********************/
	/**********************************************************************************/

	LOG(2,"Getting cubature");

	// Get numerical integration points and weights
	DefaultCubatureFactory<ST>  cubFactory;

	cubature = cubFactory.create (cellType, cubDegree);

	int cubDim       = cubature->getDimension ();
	int numCubPoints = cubature->getNumPoints ();

	cubPoints.resize(numCubPoints, cubDim);
	cubWeights.resize(numCubPoints);

	cubature->getCubature (cubPoints, cubWeights);



	/**********************************************************************************/
	/*********************************** GET BASIS ************************************/
	/**********************************************************************************/

	LOG(2,"Getting basis");
	// Define basis
	// select basis based on cell topology only for now, and assume first order basis
	switch (cellType.getKey()) {
		case shards::Tetrahedron<4>::key:
			HGradBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_TET_C1_FEM<double, Intrepid::FieldContainer<double> > );
			break;

		case shards::Hexahedron<8>::key:
			HGradBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM<double, Intrepid::FieldContainer<double> > );
			break;

		default:
			ERROR("Unknown cell topology for basis selction. Please use Hexahedron_8 or Tetrahedron_4.");
	}

	int numFieldsG = HGradBasis->getCardinality();
	HGBValues.resize(numFieldsG, numCubPoints);
	HGBGrads.resize(numFieldsG, numCubPoints, spaceDim);

	// Evaluate basis values and gradients at cubature points
	HGradBasis->getValues(HGBValues, cubPoints, OPERATOR_VALUE);
	HGradBasis->getValues(HGBGrads, cubPoints, OPERATOR_GRAD);

	/**********************************************************************************/
	/********************************* GET CUBATURE For 2D faces*********************/
	/**********************************************************************************/

	LOG(2,"Getting cubature");

	faceCubature = cubFactory.create (faceType, cubDegree);

	int faceDim       = faceCubature->getDimension ();
	int numFacePoints = faceCubature->getNumPoints ();

	facePoints.resize(numFacePoints, faceDim);
	faceWeights.resize(numFacePoints);

	faceCubature->getCubature (facePoints, faceWeights);

	/**********************************************************************************/
	/*     Get numerical integration points and weights for hexahedron face           */
	/*                  (needed for rhs boundary term)                                */
	/**********************************************************************************/

	switch (faceType.getKey()) {
		case shards::Triangle<3>::key:
			faceBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_TRI_C1_FEM<double, Intrepid::FieldContainer<double> > );
			break;

		case shards::Quadrilateral<4>::key:
			faceBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<double, Intrepid::FieldContainer<double> > );
			break;

		default:
			ERROR("Unknown face topology for basis selction. Please use Quadrilateral_4 or Triangle_3.");
	}

	int numFieldsFace = faceBasis->getCardinality();
	faceValues.resize(numFieldsFace, numFacePoints);
	HGBFaceValues.resize(numFieldsG, numFacePoints);

	// Evaluate basis values at cubature points
	faceBasis->getValues(faceValues, facePoints, OPERATOR_VALUE);
	// Evaluate HGRAD basis values at face cubature points

}

void Pde::build_maps_and_create_matrices() {

	using Teuchos::Array;
	using Teuchos::TimeMonitor;
	using Teuchos::as;
	/**********************************************************************************/
	/********************* BUILD MAPS FOR GLOBAL SOLUTION *****************************/
	/**********************************************************************************/

	LOG(2,"Building Maps");
	const int numFieldsG = HGradBasis->getCardinality();
	const int numFieldsFace = faceBasis->getCardinality();

	RCP<Teuchos::Time> timerBuildGlobalMaps =
			TimeMonitor::getNewTimer ("Build global Maps and Export");

	const long long numNodes = global_node_ids.size();
	node_on_boundary_id.resize(numNodes);
	int numBoundaryNodes = 0;
	for (int i = 0; i < numNodes; ++i) {
		if (node_on_boundary(i)) {
			node_on_boundary_id(i) = numBoundaryNodes;
			numBoundaryNodes++;
		}
	}

	RCP<const map_type> interiorMap = Tpetra::createLocalMapWithNode<LO, GO, Node>(numNodes,comm,node);
	RCP<const map_type> boundaryMap = Tpetra::createLocalMapWithNode<LO, GO, Node>(numBoundaryNodes,comm,node);
	RCP<const map_type> interiorAndboundaryMap = Tpetra::createLocalMapWithNode<LO, GO, Node>(numNodes + numBoundaryNodes,comm,node);


	/**********************************************************************************/
	/********************* BUILD GRAPHS ***********************************************/
	/**********************************************************************************/

	LOG(2,"Building Graph");

	RCP<Teuchos::Time> timerBuildOverlapGraph =
			TimeMonitor::getNewTimer ("Build graphs for overlapped and owned solutions");
	const int numElems = elem_to_node.dimension(0);
	const int numFacesPerElem = cellType.getSideCount();

		// Construct Tpetra::CrsGraph objects.
	RCP<sparse_graph_type>	interior_graph = rcp (new sparse_graph_type (interiorMap, 0));
	RCP<sparse_graph_type>	boundary_graph = rcp (new sparse_graph_type (boundaryMap, interiorMap, 0));
	RCP<sparse_graph_type>	boundary_boundary_graph = rcp (new sparse_graph_type (boundaryMap, 0));




	// Define desired workset size and count how many worksets
	// there are on this process's mesh block.
	int desiredWorksetSize = numElems; // change to desired workset size!
	//int desiredWorksetSize = 100;    // change to desired workset size!
	int numWorksets        = numElems/desiredWorksetSize;

	for (int workset = 0; workset < numWorksets; ++workset) {
		// Compute cell numbers where the workset starts and ends
		int worksetSize  = 0;
		int worksetBegin = (workset + 0)*desiredWorksetSize;
		int worksetEnd   = (workset + 1)*desiredWorksetSize;

		// When numElems is not divisible by desiredWorksetSize, the
		// last workset ends at numElems.
		worksetEnd = (worksetEnd <= numElems) ? worksetEnd : numElems;

		// Now we know the actual workset size and can allocate the
		// array for the cell nodes.
		worksetSize = worksetEnd - worksetBegin;

		//"WORKSET CELL" loop: local cell ordinal is relative to numElems
		for (int cell = worksetBegin; cell < worksetEnd; ++cell) {
			// Compute cell ordinal relative to the current workset
			//int worksetCellOrdinal = cell - worksetBegin;


			// "CELL EQUATION" loop for the workset cell: cellRow is
			// relative to the cell DoF numbering
			for (int cellRow = 0; cellRow < numFieldsG; cellRow++) {

				int localRow  = elem_to_node(cell, cellRow);
				//globalRow for Tpetra Graph
				Tpetra::global_size_t globalRowT = as<Tpetra::global_size_t> (global_node_ids[localRow]);

				// "CELL VARIABLE" loop for the workset cell: cellCol is
				// relative to the cell DoF numbering
				for (int cellCol = 0; cellCol < numFieldsG; ++cellCol) {
					int localCol  = elem_to_node (cell, cellCol);
					int globalCol = as<int> (global_node_ids[localCol]);
					//create ArrayView globalCol object for Tpetra
					Teuchos::ArrayView<int> globalColAV = Teuchos::arrayView (&globalCol, 1);

					//Update Tpetra overlap Graph
					interior_graph->insertGlobalIndices (globalRowT, globalColAV);
				}// *** cell col loop ***



			}// *** cell row loop ***
		}// *** workset cell loop **
	}// *** workset loop ***


	// add node -> boundary node connectivity
	const int num_boundary_faces = boundary_face_to_elem.dimension(0);
	for (int i = 0; i < num_boundary_faces; ++i) {
		const int iface = boundary_face_to_ordinal(i);
		const int ielem = boundary_face_to_elem(i);
		for (int iface_point = 0; iface_point < numNodesPerFace; ++iface_point) {
			const int sideNode = cellType.getNodeMap(2,iface,iface_point);
			const int local_index_bp = elem_to_node(ielem,sideNode);
			Tpetra::global_size_t global_index_bcp = as<Tpetra::global_size_t> (node_on_boundary_id(local_index_bp));

			int global_index_bcp_int = as<int> (global_index_bcp);
			Teuchos::ArrayView<int> global_index_bcp_AV = Teuchos::arrayView (&global_index_bcp_int, 1);
			for (int cellpt = 0; cellpt < numNodesPerFace; cellpt++) {
				const int sideNode2 = cellType.getNodeMap(2,iface,cellpt);
				const int local_index_p  = elem_to_node(ielem, sideNode2);
				//globalRow for Tpetra Graph
				Tpetra::global_size_t global_index_p = as<Tpetra::global_size_t> (global_node_ids[local_index_p]);
				Tpetra::global_size_t global_index_bcp2 = as<Tpetra::global_size_t> (node_on_boundary_id(local_index_p));

				int global_index_p_int = as<int> (global_index_p);
				Teuchos::ArrayView<int> global_index_p_AV = Teuchos::arrayView (&global_index_p_int, 1);

				int global_index_bcp2_int = as<int> (global_index_bcp2);
				Teuchos::ArrayView<int> global_index_bcp2_AV = Teuchos::arrayView (&global_index_bcp2_int, 1);
				boundary_graph->insertGlobalIndices (global_index_bcp, global_index_p_AV);
				boundary_boundary_graph->insertGlobalIndices (global_index_bcp, global_index_bcp2_AV);

			}
		}
	}

	// Fill-complete overlapping distribution Graph.
	interior_graph->fillComplete ();
	boundary_graph->fillComplete(interiorMap,boundaryMap);
	boundary_boundary_graph->fillComplete();


	LOG(2,"Constructing LHS and RHS matrix and vectors");

	//
	// Construct stiffness matrix, right-hand side vector, and exact
	// solution vector.  The linear algebra objects starting with gl_
	// are for the nonoverlapped ("owned") distribution; the ones not
	// starting with gl_ are for the overlapped distribution.  Once
	// we've constructed the overlapped distribution objects, we'll
	// Export to the owned distribution objects.
	//
	// Owned distribution objects:
	//
	K = rcp (new sparse_matrix_type (interior_graph.getConst ()));
	Mi = rcp (new sparse_matrix_type (interior_graph.getConst ()));
	B = rcp (new sparse_matrix_type (boundary_graph.getConst ()));
	Mb = rcp (new sparse_matrix_type (boundary_boundary_graph.getConst ()));

	U_Lambda = rcp (new vector_type (interiorAndboundaryMap));
	U_Lambda_rhs = rcp (new vector_type (interiorAndboundaryMap));
	U_Lambda->putScalar(0);
	U_Lambda_rhs->putScalar(0);

	U = U_Lambda->offsetViewNonConst(interiorMap,0)->getVectorNonConst(0);
	Lambda = U_Lambda->offsetViewNonConst(boundaryMap,numNodes)->getVectorNonConst(0);
	U_rhs = U_Lambda_rhs->offsetViewNonConst(interiorMap,0)->getVectorNonConst(0);
	Lambda_rhs = U_Lambda_rhs->offsetViewNonConst(boundaryMap,numNodes)->getVectorNonConst(0);

//	U = rcp (new vector_type (interiorMap));
//	Lambda = rcp (new vector_type (boundaryMap));
//	U->putScalar(0);
//	Lambda->putScalar(0);
//
//	U_rhs = rcp (new vector_type (interiorMap));
//	Lambda_rhs = rcp (new vector_type (boundaryMap));
//	U_rhs->putScalar(0);
//	Lambda_rhs->putScalar(0);

	boundary_node_areas = rcp (new vector_type (boundaryMap));
	interior_node_volumes = rcp (new vector_type (interiorMap));


	boundary_node_positions = rcp (new multivector_type (boundaryMap,spaceDim));

	int ownedBoundaryNodes = 0;
	for (int i = 0; i < numNodes; ++i) {
		if (node_is_owned[i]) {
			if (node_on_boundary(i)) {
				for (size_t j = 0; j < spaceDim; ++j) {
					boundary_node_positions->replaceLocalValue(ownedBoundaryNodes,j,node_coord(i,j));
				}
				++ownedBoundaryNodes;
			}
		}
	}
}

void Pde::integrate(const ST requested_dt) {

	Teuchos::RCP<Teuchos::Time> timer =
			Teuchos::TimeMonitor::getNewTimer ("Total Pde Integration");
	Teuchos::TimeMonitor timerMon (*timer);
	const int iterations = int(requested_dt/dt + 0.5);
	const double actual_dt = iterations*dt;
	std::cout << "integrating for "<<actual_dt<<" seconds (" << iterations << " iterations)" << std::endl;
	for (int i = 0; i < iterations; ++i) {
		/*
		 * Solve FEM system
		 */
		solve();
	}

}


void Pde::fillMatricies () {

	using Teuchos::TimeMonitor;
	using Teuchos::ArrayView;

	typedef Teuchos::ScalarTraits<ST> STS;


	RCP<Teuchos::Time> timerAssembleBoundaryIntegral =
			TimeMonitor::getNewTimer ("Assemble Boundary Integral");
	{
		TimeMonitor timerAssembleGlobalMatrixL (*timerAssembleBoundaryIntegral);

		boundary_integrals();
	}

	RCP<Teuchos::Time> timerAssembleVolumeIntegral =
			TimeMonitor::getNewTimer ("Assemble Volume Integral");
	{
		TimeMonitor timerAssembleGlobalMatrixL (*timerAssembleVolumeIntegral);

		volume_integrals();
	}

}


template<typename Scalar>
void Pde::materialTensor (Scalar material[][3], const Scalar& x, const Scalar& y,const Scalar& z) {
	typedef Teuchos::ScalarTraits<Scalar> STS;

	material[0][0] = STS::one();
	material[0][1] = STS::zero();
	material[0][2] = STS::zero();

	material[1][0] = STS::zero();
	material[1][1] = STS::one();
	material[1][2] = STS::zero();

	material[2][0] = STS::zero();
	material[2][1] = STS::zero();
	material[2][2] = STS::one();

}


//! Compute the material tensor over a workset.
template<class ArrayOut, class ArrayIn>
void Pde::evaluateMaterialTensor (ArrayOut& matTensorValues, const ArrayIn& evaluationPoints) {
  typedef typename ArrayOut::scalar_type scalar_type;

  const int numWorksetCells  = evaluationPoints.dimension(0);
  const int numPoints        = evaluationPoints.dimension(1);
  const int spaceDim         = evaluationPoints.dimension(2);

  scalar_type material[3][3];

  for (int cell = 0; cell < numWorksetCells; ++cell) {
    for (int pt = 0; pt < numPoints; ++pt) {
      scalar_type x = evaluationPoints(cell, pt, 0);
      scalar_type y = evaluationPoints(cell, pt, 1);
      scalar_type z = evaluationPoints(cell, pt, 2);

      materialTensor<scalar_type> (material, x, y, z);

      for (int row = 0; row < spaceDim; ++row) {
        for(int col = 0; col < spaceDim; ++col) {
          matTensorValues(cell, pt, row, col) = material[row][col];
        }
      }
    }
  }
}





void Pde::solve() {
	typedef Teuchos::ScalarTraits<ST> STS;
	typedef STS::magnitudeType MT;
	typedef Teuchos::ScalarTraits<MT> STM;


	RHS->apply(Thyra::NOTRANS, *X.getConst(), Y.ptr(),1,0);

	//Thyra::SolveStatus<ST>
	//	    status = LHS->solve(Thyra::NOTRANS, *Y.getConst(), X.ptr());

	RCP<const Thyra::LinearOpBase<ST> > A = dynamic_cast<Thyra::BlockedLinearOpBase>(LHS)->getBlock(0,0);
	RCP<const Thyra::LinearOpBase<ST> > B = LHS->getBlock(0,1);
	RCP<const Thyra::LinearOpBase<ST> > C = LHS->getBlock(1,0);
	RCP<const Thyra::LinearOpBase<ST> > D = LHS->getBlock(1,1);
	RCP<Thyra::VectorBase<ST> > u = X->getBlock(0);
	RCP<Thyra::VectorBase<ST> > lambda = X->getBlock(1);
	RCP<Thyra::VectorBase<ST> > u_rhs = Y->getBlock(0);
	RCP<Thyra::VectorBase<ST> > lambda_rhs = Y->getBlock(1);

	RCP<const Thyra::LinearOpBase<ST> > C_invA = Thyra::multiply(C,invA);
	C_invA->apply(Thyra::NOTRANS, *u_rhs.getConst(), lambda_rhs.ptr(),-1,1);  //l_r = l_r - C_invA*u_r

	
	Thyra::SolveStatus<ST>
		    status = S->solve(Thyra::NOTRANS, *lambda_rhs.getConst(), lambda.ptr()); //l = invS*l_r

	std::cout << "S solve: "<< status.message << std::endl;

	B->apply(Thyra::NOTRANS, *lambda.getConst(), u_rhs.ptr(),-1,1);      //u_r = u_r - B*lambda

	invA->apply(Thyra::NOTRANS, *u_rhs.getConst(), u.ptr(),1,0);         //u = invA*u_r

	//Thyra::SolveStatus<ST>
	//	    status = A->solve(Thyra::NOTRANS, *u_rhs.getConst(), u.ptr());

	//std::cout << "S solve: "<< status.message << std::endl;
}


std::string Pde::makeMeshInput (const int nx, const int ny, const int nz) {
  using std::endl;
  std::ostringstream os;

  TEUCHOS_TEST_FOR_EXCEPTION( nx <= 0 || ny <= 0 || nz <= 0,
    std::invalid_argument, "nx, ny, and nz must all be positive.");

  os << "mesh" << endl
     << "\trectilinear" << endl
     << "\t\tnx = " << nx << endl
     << "\t\tny = " << ny << endl
     << "\t\tnz = " << nz << endl
     << "\t\tbx = 1" << endl
     << "\t\tby = 1" << endl
     << "\t\tbz = 1" << endl
     << "\t\tgmin = 0 0 0" << endl
     << "\t\tgmax = 1 1 1" << endl
     << "\tend" << endl
     << "\tset assign" << endl
//     << "\t\tsideset, ilo, 1" << endl
//     << "\t\tsideset, jlo, 2" << endl
//     << "\t\tsideset, klo, 3" << endl
     << "\t\tsideset, ihi, 4" << endl
//     << "\t\tsideset, jhi, 5" << endl
//     << "\t\tsideset, khi, 6" << endl
     << "\tend" << endl
     << "end";
  return os.str ();
}

std::string Pde::makeMeshInputFullDomain (const int nx, const int ny, const int nz) {
  using std::endl;
  std::ostringstream os;

  TEUCHOS_TEST_FOR_EXCEPTION( nx <= 0 || ny <= 0 || nz <= 0,
    std::invalid_argument, "nx, ny, and nz must all be positive.");

  os << "mesh" << endl
     << "\trectilinear" << endl
     << "\t\tnx = " << nx << endl
     << "\t\tny = " << ny << endl
     << "\t\tnz = " << nz << endl
     << "\t\tbx = 1" << endl
     << "\t\tby = 1" << endl
     << "\t\tbz = 1" << endl
     << "\t\tgmin = 0 0 0" << endl
     << "\t\tgmax = 2 1 1" << endl
     << "\tend" << endl
     << "end";
  return os.str ();
}

std::string Pde::makeMeshInputSphere (const int nr, const int ntheta) {
  using std::endl;
  std::ostringstream os;

  os << "mesh" << endl
     << "\tspherical" << endl
     << "\t\tri = " << ri << endl
     << "\t\tro = " << ro << endl
     << "\t\tntheta =" << ntheta << endl
     << "\t\tnphi =" << ntheta << endl
     << "\t\tnr = " << nr << endl
     << "\t\tbr	 = 1" << endl
     << "\t\tbtheta = 1" << endl
     << "\t\tbphi = 1" << endl
     << "\t\ttheta = 90" << endl
     << "\t\tphi = 90" << endl
     << "\tend" << endl
     << "\tset assign" << endl
     << "\t\tsideset, ilo, 1" << endl


     << "\t\tsideset, ihi, 4" << endl

     << "\tend" << endl
     << "end";
  return os.str ();
}

std::string Pde::makeMeshInputRadialTrisection (const int nr, const int ntheta, const int nz) {
  using std::endl;
  std::ostringstream os;

  os << "mesh" << endl
     << "\tradial trisection" << endl
     << "\t\ttrisection blocks, 3" << endl
     << "\t\ttransition radius, 6.5" << endl
     << "\tnumz 1"<< endl
     << "\t\tzblock 1 1. interval "<<nz<<endl
     << "\tnumr 2"<< endl
     << "\t\trblock 1 "<<0.2*ro<<" interval "<<int(0.2*nr)<<endl
     << "\t\trblock 2 "<<0.8*ro<<" interval "<<int(0.8*nr)<<endl
     << "\tnuma 1"<< endl
     << "\t\tablock 1 90. interval "<<ntheta<<endl
     << "\tend" << endl
     << "\tset assign" << endl
     << "\t\tsideset, ihi, 1" << endl
     << "\tend" << endl
     << "end";
  return os.str ();
}

std::string Pde::makeMeshInputCylinder (const int nr, const int ntheta, const int nz) {
  using std::endl;
  std::ostringstream os;

  os << "mesh" << endl
     << "\tcylindrical" << endl
     << "\t\tri = " << ri << endl
     << "\t\tro = " << ro << endl
     << "\t\tnr = " << nr << endl
     << "\t\tbr	 = 1" << endl
     << "\t\tntheta =" << ntheta << endl
     << "\t\tbtheta = 1" << endl
     << "\t\ttheta = 90" << endl
     << "\t\tzmin = 0" << endl
     << "\t\tzmax = 1" << endl
     << "\t\tnz = " << nz << endl
     << "\t\tbz	 = 1" << endl
     << "\tend" << endl
     << "\tset assign" << endl
     << "\t\tsideset, ilo, 1" << endl
     << "\t\tsideset, ihi, 2" << endl
     << "\tend" << endl
     << "end";
  return os.str ();

}

vtkUnstructuredGrid* Pde::get_grid() {
	return vtk_grid;
}

vtkUnstructuredGrid* Pde::get_boundary() {
	return vtk_boundary;
}

void Pde::create_vtk_grid() {
	//TODO: assumes 1 cpu
	/*
	 * setup points
	 */
	vtkSmartPointer<vtkPoints> newPts = vtkSmartPointer<vtkPoints>::New();
	const int num_overlapped_points = node_coord.dimension(0);
	for (int i = 0; i < num_overlapped_points; i++) {
		newPts->InsertNextPoint(node_coord(i,0),node_coord(i,1),node_coord(i,2));
	}

	/*
	 * setup scalar data
	 */
	vtkSmartPointer<vtkDoubleArray> newScalars = vtkSmartPointer<vtkDoubleArray>::New();
	const int num_local_entries = U->getLocalLength();
	newScalars->SetArray(U->getDataNonConst(0).getRawPtr(),num_overlapped_points,1);
	newScalars->SetName("Concentration");



	/*
	 * setup cells
	 */
	const int num_cells = elem_to_node.dimension(0);
	vtk_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
	vtk_grid->Allocate(num_cells,num_cells);
	vtk_grid->SetPoints(newPts);
	vtk_grid->GetPointData()->SetScalars(newScalars);
	const int num_points_per_cell = elem_to_node.dimension(1);
	for (int i = 0; i < num_cells; ++i) {
		vtkSmartPointer<vtkHexahedron> newHex = vtkSmartPointer<vtkHexahedron>::New();
		newHex->GetPointIds()-> SetNumberOfIds(num_points_per_cell);
		for (int j = 0; j < num_points_per_cell; ++j) {
			newHex->GetPointIds()-> SetId(j,elem_to_node(i,j));
		}
		vtk_grid->InsertNextCell(newHex->GetCellType(),newHex->GetPointIds());

	}


	/*
	 * setup boundary points
	 */
	vtkSmartPointer<vtkPoints> boundary_Pts = vtkSmartPointer<vtkPoints>::New();
	int numBoundaryPoints = 0;
	for (int i = 0; i < num_overlapped_points; i++) {
		if (node_on_boundary(i)) {
			boundary_Pts->InsertNextPoint(node_coord(i,0),node_coord(i,1),node_coord(i,2));
			numBoundaryPoints++;
		}
	}

	/*
	 * setup Outflow data
	 */
	vtkSmartPointer<vtkDoubleArray> outflow_scalar = vtkSmartPointer<vtkDoubleArray>::New();
	outflow_scalar->SetArray(Lambda->getDataNonConst(0).getRawPtr(),Lambda->getLocalLength(),1);
	outflow_scalar->SetName("Outflow");

	/*
	 * setup boundary faces
	 */
	const int num_faces = boundary_face_to_elem.dimension(0);
	vtk_boundary = vtkSmartPointer<vtkUnstructuredGrid>::New();
	vtk_boundary->Allocate(num_faces,num_faces);
	vtk_boundary->SetPoints(boundary_Pts);
	vtk_boundary->GetPointData()->SetScalars(outflow_scalar);
	const int num_nodes_per_face = faceType.getNodeCount();
	for (int i = 0; i < num_faces; ++i) {
		const int iface = boundary_face_to_ordinal(i);
		const int ielem = boundary_face_to_elem(i);
		vtkSmartPointer<vtkQuad> newQuad = vtkSmartPointer<vtkQuad>::New();
		newQuad->GetPointIds()-> SetNumberOfIds(num_nodes_per_face);
		for (int j = 0; j < num_nodes_per_face; ++j) {
			const int sideNode = cellType.getNodeMap(2,iface,j);
			const int local_index_bp = elem_to_node(ielem,sideNode);
			const int local_index_bcnodes = node_on_boundary_id(local_index_bp);
			newQuad->GetPointIds()-> SetId(j,local_index_bcnodes);
			if (i==0) std::cout << "setting node id = "<<local_index_bcnodes<<std::endl;
		}
		vtk_boundary->InsertNextCell(newQuad->GetCellType(),newQuad->GetPointIds());

	}
}



RCP<Pde::vector_type> Pde::get_boundary_node_values() {
	return Lambda;
}

RCP<Pde::vector_type> Pde::get_boundary_node_areas() {
	return boundary_node_areas;
}

RCP<Pde::multivector_type> Pde::get_boundary_node_positions() {
	return boundary_node_positions;
}

int Pde::get_total_number_of_particles() {
	using Tpetra::RTI::reduce;
	using Tpetra::RTI::ZeroOp;
	using Tpetra::RTI::reductionGlob;

	RCP<vector_type> M_times_concentrations =
			rcp (new vector_type (U->getMap(), true));
	Mi->apply(*U.getConst(),*M_times_concentrations);
	return TPETRA_REDUCE1(M_times_concentrations, M_times_concentrations, ZeroOp<double>, std::plus<double>());
}

void Pde::calculate_volumes_and_areas() {
	using Tpetra::RTI::reduce;
	using Tpetra::RTI::ZeroOp;
	using Tpetra::RTI::reductionGlob;

	RCP<vector_type> ones =
				rcp (new vector_type (U->getMap(), false));
	ones->putScalar(1.0);


	Mi->apply(*ones.getConst(),*interior_node_volumes);
	std::cout << "total volume is " <<
			TPETRA_REDUCE1(interior_node_volumes, interior_node_volumes, ZeroOp<double>, std::plus<double>()) << std::endl;


	RCP<vector_type> onesL =
			rcp (new vector_type (Lambda->getMap(), false));
	onesL->putScalar(1.0);


	Mb->apply(*onesL.getConst(),*boundary_node_areas);

	std::cout << "total boundary area is " <<
			TPETRA_REDUCE1(boundary_node_areas, boundary_node_areas, ZeroOp<double>, std::plus<double>()) << std::endl;
}

void Pde::rescale(double s) {
	U->scale(s);
}


void Pde::boundary_integrals() {
	using namespace Intrepid;
	using Teuchos::ArrayView;
	using Teuchos::arrayView;
	using Teuchos::as;
	typedef Teuchos::ScalarTraits<ST> STS;

	using Teuchos::TimeMonitor;
	typedef Intrepid::FunctionSpaceTools IntrepidFSTools;
	typedef Intrepid::RealSpaceTools<ST> IntrepidRSTools;
	typedef Intrepid::CellTools<ST>      IntrepidCTools;

	const int numBoundaryFaces = boundary_face_to_elem.dimension(0);
	const int numNodesPerFace = faceType.getNodeCount();
	const int numFieldsG = HGradBasis->getCardinality();
	const int numFieldsFace = faceBasis->getCardinality();
	const int numCubPoints = faceCubature->getNumPoints();
	const int cubDim = cubature->getDimension();


	B->setAllToScalar (STS::zero ());
	Mb->setAllToScalar (STS::zero ());

	/**********************************************************************************/
	/******************** DEFINE WORKSETS AND LOOP OVER THEM **************************/
	/**********************************************************************************/
	if (numBoundaryFaces == 0) return;
	// Define desired workset size and count how many worksets there are
	// on this processor's mesh block
	int desiredWorksetSize = numBoundaryFaces; // change to desired workset size!
	//int desiredWorksetSize = 100;    // change to desired workset size!
	int numWorksets        = numBoundaryFaces/desiredWorksetSize;

	// When numElems is not divisible by desiredWorksetSize, increase
	// workset count by 1
	if (numWorksets*desiredWorksetSize < numBoundaryFaces) {
		numWorksets += 1;
	}

	LOG(2,"Desired workset size:             " << desiredWorksetSize << std::endl
			<< "Number of worksets (per process): " << numWorksets);

	for (int workset = 0; workset < numWorksets; ++workset) {
		// Compute cell numbers where the workset starts and ends
		int worksetSize  = 0;
		int worksetBegin = (workset + 0)*desiredWorksetSize;
		int worksetEnd   = (workset + 1)*desiredWorksetSize;

		// When numElems is not divisible by desiredWorksetSize, the last
		// workset ends at numElems.
		worksetEnd = (worksetEnd <= numBoundaryFaces) ? worksetEnd : numBoundaryFaces;

		// Now we know the actual workset size and can allocate the array
		// for the cell nodes.
		worksetSize = worksetEnd - worksetBegin;
		FieldContainer<ST> cellWorkset (1, numFieldsG, spaceDim);
		FieldContainer<ST> worksetCubWeights(1, numCubPoints);

		FieldContainer<ST> worksetRefCubPoints (numCubPoints, spaceDim);
		// Copy coordinates and face cubature points (in the ref cell domain)
		// into cell workset
		FieldContainer<ST> worksetJacobian  (1, numCubPoints, spaceDim, spaceDim);

		FieldContainer<ST> worksetFaceValuesWeighted(1, numFieldsFace, numCubPoints);
		FieldContainer<ST> worksetFaceValues        (1, numFieldsFace, numCubPoints);

		// Containers for workset contributions to the boundary integral
		FieldContainer<ST> worksetWeakBC (1, numFieldsFace, numFieldsFace);

		for (int face = worksetBegin; face < worksetEnd; ++face) {
			const int ielem = boundary_face_to_elem(face);
			const int iface = boundary_face_to_ordinal(face);
			for (int node = 0; node < numFieldsG; ++node) {
				const int node_num = elem_to_node(ielem, node);
				for (int j = 0; j < spaceDim; ++j) {
					cellWorkset(0, node, j) = node_coord(node_num, j);
				}
			}


			IntrepidCTools::mapToReferenceSubcell(worksetRefCubPoints,
					facePoints,
					2, iface, cellType);


			IntrepidCTools::setJacobian(worksetJacobian, worksetRefCubPoints,
							cellWorkset, cellType);

			IntrepidFSTools::computeFaceMeasure<ST> (worksetCubWeights, // Det(DF)*w = J*w
							worksetJacobian,
							faceWeights,
							iface,
							cellType);

			IntrepidFSTools::HGRADtransformVALUE<ST> (worksetFaceValues, // clones basis values (mu)
					faceValues);

			IntrepidFSTools::multiplyMeasure<ST> (worksetFaceValuesWeighted, // (u)*w
					worksetCubWeights,
					worksetFaceValues);

			// Integrate to compute workset contribution to global matrix:
			IntrepidFSTools::integrate<ST> (worksetWeakBC, // (u)*(u)*w
					worksetFaceValues,
					worksetFaceValuesWeighted,
					COMP_BLAS);


			for (int face_pt = 0; face_pt < numFieldsFace; ++face_pt) {
				const int sideNode = cellType.getNodeMap(2,iface,face_pt);
				const LO local_face_pt_id = elem_to_node(ielem,sideNode);
				GO global_face_pt_id =
						as<int> (node_on_boundary_id(local_face_pt_id));
				ArrayView<GO> global_face_pt_AV = arrayView<GO> (&global_face_pt_id, 1);
				// "CELL EQUATION" loop for the workset cell: cellRow is
				// relative to the cell DoF numbering.
				for (int cell_pt = 0; cell_pt < numFieldsFace; ++cell_pt) {
					const int sideNode2 = cellType.getNodeMap(2,iface,cell_pt);
					const LO local_cell_pt_id = elem_to_node(ielem,sideNode2);
					GO global_cell_pt_id = as<GO> (global_node_ids[local_cell_pt_id]);
					GO global_face_pt_id2 = as<GO> (node_on_boundary_id[local_cell_pt_id]);

					ArrayView<GO> global_cell_pt_AV = arrayView<GO> (&global_cell_pt_id, 1);
					ArrayView<GO> global_face_pt_AV2 = arrayView<GO> (&global_face_pt_id2, 1);

					ST operatorBContribution =
							worksetWeakBC (0, face_pt, cell_pt);

					B->sumIntoGlobalValues (global_face_pt_id, global_cell_pt_AV,
							arrayView<ST> (&operatorBContribution, 1));
					Mb->sumIntoGlobalValues (global_face_pt_id, global_face_pt_AV2,
							arrayView<ST> (&operatorBContribution, 1));


				}// *** cell col loop ***
			}// *** cell row loop ***
		}// *** workset cell loop **
	}// *** workset loop ***

	B->fillComplete(B->getColMap(),B->getRangeMap());
	Mb->fillComplete();
}

void Pde::volume_integrals() {
	using namespace Intrepid;
	using Teuchos::ArrayView;
	using Teuchos::arrayView;
	using Teuchos::as;
	typedef Teuchos::ScalarTraits<ST> STS;

	using Teuchos::TimeMonitor;

	typedef Intrepid::FunctionSpaceTools IntrepidFSTools;
	typedef Intrepid::RealSpaceTools<ST> IntrepidRSTools;
	typedef Intrepid::CellTools<ST>      IntrepidCTools;

	const int numFieldsG = HGradBasis->getCardinality();
	const int numCubPoints = cubature->getNumPoints();
	const int numElems = elem_to_node.dimension(0);
	const long long numNodes = global_node_ids.size();
	const int cubDim = cubature->getDimension();

	Mi->setAllToScalar (STS::zero ());
	K->setAllToScalar (STS::zero ());

	/**********************************************************************************/
	/******************** DEFINE WORKSETS AND LOOP OVER THEM **************************/
	/**********************************************************************************/

	LOG(2,"Building discretization matricies");


	// Define desired workset size and count how many worksets there are
	// on this processor's mesh block
	int desiredWorksetSize = numElems; // change to desired workset size!
	//int desiredWorksetSize = 100;    // change to desired workset size!
	int numWorksets        = numElems/desiredWorksetSize;

	// When numElems is not divisible by desiredWorksetSize, increase
	// workset count by 1
	if (numWorksets*desiredWorksetSize < numElems) {
		numWorksets += 1;
	}

	LOG(2,"Desired workset size:             " << desiredWorksetSize << std::endl
			<< "Number of worksets (per process): " << numWorksets);

	for (int workset = 0; workset < numWorksets; ++workset) {
		// Compute cell numbers where the workset starts and ends
		int worksetSize  = 0;
		int worksetBegin = (workset + 0)*desiredWorksetSize;
		int worksetEnd   = (workset + 1)*desiredWorksetSize;

		// When numElems is not divisible by desiredWorksetSize, the last
		// workset ends at numElems.
		worksetEnd = (worksetEnd <= numElems) ? worksetEnd : numElems;

		// Now we know the actual workset size and can allocate the array
		// for the cell nodes.
		worksetSize = worksetEnd - worksetBegin;
		FieldContainer<ST> cellWorkset (worksetSize, numFieldsG, spaceDim);

		// array to contain boundary normals (=0 if not on boundary)
		FieldContainer<ST> boundary_normals(worksetSize, spaceDim);

		// Copy coordinates into cell workset
		int cellCounter = 0;
		for (int cell = worksetBegin; cell < worksetEnd; ++cell) {
			for (int node = 0; node < numFieldsG; ++node) {
				const int node_num = elem_to_node(cell, node);
				cellWorkset(cellCounter, node, 0) = node_coord(node_num, 0);
				cellWorkset(cellCounter, node, 1) = node_coord(node_num, 1);
				cellWorkset(cellCounter, node, 2) = node_coord(node_num, 2);

			}
			++cellCounter;
		}

		/**********************************************************************************/
		/*                                Allocate arrays                                 */
		/**********************************************************************************/

		// Containers for Jacobians, integration measure & cubature points in workset cells
		FieldContainer<ST> worksetJacobian  (worksetSize, numCubPoints, spaceDim, spaceDim);
		FieldContainer<ST> worksetJacobInv  (worksetSize, numCubPoints, spaceDim, spaceDim);
		FieldContainer<ST> worksetJacobDet  (worksetSize, numCubPoints);
		FieldContainer<ST> worksetCubWeights(worksetSize, numCubPoints);
		FieldContainer<ST> worksetCubPoints (worksetSize, numCubPoints, cubDim);

		// Containers for basis values transformed to workset cells and
		// them multiplied by cubature weights
		FieldContainer<ST> worksetHGBValues        (worksetSize, numFieldsG, numCubPoints);
		FieldContainer<ST> worksetHGBValuesWeighted(worksetSize, numFieldsG, numCubPoints);
		FieldContainer<ST> worksetHGBGrads         (worksetSize, numFieldsG, numCubPoints, spaceDim);
		FieldContainer<ST> worksetHGBGradsWeighted (worksetSize, numFieldsG, numCubPoints, spaceDim);

		// Additional arrays used in analytic assembly
		//		FieldContainer<ST> u_coeffs(worksetSize, numFieldsG);
		//		FieldContainer<ST> u_FE_val(worksetSize, numCubPoints);
		//		FieldContainer<ST> df_of_u(worksetSize, numCubPoints);
		//		FieldContainer<ST> df_of_u_times_basis(worksetSize, numFieldsG, numCubPoints);

		// Containers for diffusive & advective fluxes & non-conservative
		// adv. term and reactive terms
		FieldContainer<ST> worksetDiffusiveFlux(worksetSize, numFieldsG, numCubPoints, spaceDim);

		// Containers for material values and source term. Require
		// user-defined functions
		FieldContainer<ST> worksetMaterialVals (worksetSize, numCubPoints, spaceDim, spaceDim);

		// Containers for workset contributions to the discretization
		// matrix and the right hand side
		FieldContainer<ST> worksetStiffMatrix (worksetSize, numFieldsG, numFieldsG);
		FieldContainer<ST> worksetMassMatrix (worksetSize, numFieldsG, numFieldsG);
		FieldContainer<ST> worksetGradOp (worksetSize, numFieldsG, numFieldsG);


		/**********************************************************************************/
		/*                                Calculate Jacobians                             */
		/**********************************************************************************/

		IntrepidCTools::setJacobian(worksetJacobian, cubPoints, cellWorkset, cellType);
		IntrepidCTools::setJacobianInv(worksetJacobInv, worksetJacobian );
		IntrepidCTools::setJacobianDet(worksetJacobDet, worksetJacobian );

		/**********************************************************************************/
		/*          Cubature Points to Physical Frame and Compute Data                    */
		/**********************************************************************************/

		// Map cubature points to physical frame.
		IntrepidCTools::mapToPhysicalFrame (worksetCubPoints, cubPoints, cellWorkset, cellType);

		// Evaluate the material tensor A at cubature points.
		evaluateMaterialTensor (worksetMaterialVals, worksetCubPoints);

		/**********************************************************************************/
		/*                         Compute Stiffness Matrix                               */
		/**********************************************************************************/

		// Transform basis gradients to physical frame:
		IntrepidFSTools::HGRADtransformGRAD<ST> (worksetHGBGrads,   // DF^{-T}(grad u)
				worksetJacobInv,
				HGBGrads);
		// Compute integration measure for workset cells:
		IntrepidFSTools::computeCellMeasure<ST> (worksetCubWeights, // Det(DF)*w = J*w
				worksetJacobDet,
				cubWeights);
		// Multiply transformed (workset) gradients with weighted measure
		IntrepidFSTools::multiplyMeasure<ST> (worksetHGBGradsWeighted, // DF^{-T}(grad u)*J*w
				worksetCubWeights,
				worksetHGBGrads);
		// Compute the diffusive flux:
		IntrepidFSTools::tensorMultiplyDataField<ST> (worksetDiffusiveFlux, // A*(DF^{-T}(grad u)
				worksetMaterialVals,
				worksetHGBGrads);
		// Integrate to compute workset diffusion contribution to global matrix:
		IntrepidFSTools::integrate<ST> (worksetStiffMatrix, // (DF^{-T}(grad u)*J*w)*(A*DF^{-T}(grad u))
				worksetHGBGradsWeighted,
				worksetDiffusiveFlux,
				COMP_BLAS);

		/**********************************************************************************/
		/*                         Compute Mass Matrix                               */
		/**********************************************************************************/


		//Transform basis values to physical frame:
		IntrepidFSTools::HGRADtransformVALUE<ST> (worksetHGBValues, // clones basis values (u)
				HGBValues);
		// Multiply transformed (workset) gradients with weighted measure
		IntrepidFSTools::multiplyMeasure<ST> (worksetHGBValuesWeighted, // (u)*w
				worksetCubWeights,
				worksetHGBValues);
		// Integrate to compute workset contribution to global matrix:
		IntrepidFSTools::integrate<ST> (worksetMassMatrix, // (u)*(u)*w
				worksetHGBValues,
				worksetHGBValuesWeighted,
				COMP_BLAS);

		//		//Mass Lumped?????
		//		//Transform basis values to physical frame:
		//		IntrepidFSTools::HGRADtransformVALUE<ST> (worksetHGBValues, // clones basis values (u)
		//				HGBValues);
		//		// Multiply transformed (workset) gradients with weighted measure
		//		cubWeights.resize(1,numCubPoints);
		//		IntrepidFSTools::multiplyMeasure<ST> (worksetHGBValuesWeighted, // (u)*w
		//				worksetCubWeights,
		//				worksetHGBValues);
		//		cubWeights.resize(numCubPoints);
		//		// Integrate to compute workset contribution to global matrix:
		//		IntrepidFSTools::integrate<ST> (worksetMassMatrix, // (u)*(u)*w
		//				worksetHGBValues,
		//				worksetHGBValuesWeighted,
		//				COMP_BLAS);


		//		/**********************************************************************************/
		//		/*                                   Compute Reaction                             */
		//		/**********************************************************************************/
		//
		//
		//		// Transform basis values to physical frame:
		//		IntrepidFSTools::HGRADtransformVALUE<ST> (worksetHGBValues, // clones basis values (u)
		//				HGBValues);
		//		// Multiply transformed (workset) values with weighted measure
		//		IntrepidFSTools::multiplyMeasure<ST> (worksetHGBValuesWeighted, // (u)*J*w
		//				worksetCubWeights,
		//				worksetHGBValues);
		//		// Integrate worksetSourceTerm against weighted basis function set
		//		IntrepidFSTools::integrate<ST> (worksetSource, // f.(u)*J*w
		//				worksetSourceTerm,
		//				worksetHGBValuesWeighted,
		//				COMP_BLAS);
		//
		//		      // u_coeffs equals the value of u_coeffsAD
		//		      for(int i=0; i<numFieldsG; i++){
		//		        u_coeffs(0,i) = u_coeffsAD(0,i).val();
		//		      }
		//		      // represent value of the current state (iterate) as a linear combination of the basis functions
		//		      u_FE_val.initialize();
		//		      fst::evaluate<double>(u_FE_val, u_coeffs, hexGValsTransformed);
		//
		//		      // evaluate derivative of the nonlinear term and multiply by basis function
		//		      dfunc_u(df_of_u, u_FE_val);
		//		      fst::scalarMultiplyDataField<double>(df_of_u_times_basis, df_of_u, hexGValsTransformed);
		//
		//		      // integrate to account for nonlinear reaction term
		//		      fst::integrate<double>(localPDEjacobian, df_of_u_times_basis, hexGValsTransformedWeighted, INTREPID_INTEGRATE_COMP_ENGINE, true);

		//		/**********************************************************************************/
		//		/*                                   Compute Source                               */
		//		/**********************************************************************************/
		//
		//		// Transform basis values to physical frame:
		//		IntrepidFSTools::HGRADtransformVALUE<ST> (worksetHGBValues, // clones basis values (u)
		//				HGBValues);
		//		// Multiply transformed (workset) values with weighted measure
		//		IntrepidFSTools::multiplyMeasure<ST> (worksetHGBValuesWeighted, // (u)*J*w
		//				worksetCubWeights,
		//				worksetHGBValues);
		//		// Integrate worksetSourceTerm against weighted basis function set
		//		IntrepidFSTools::integrate<ST> (worksetSource, // f.(u)*J*w
		//				worksetSourceTerm,
		//				worksetHGBValuesWeighted,
		//				COMP_BLAS);

		/**********************************************************************************/
		/*                         Assemble into Global Matrix                            */
		/**********************************************************************************/

		RCP<Teuchos::Time> timerAssembleGlobalMatrix =
				TimeMonitor::getNewTimer ("Assemble overlapped global matrix and Source");
		{
			TimeMonitor timerAssembleGlobalMatrixL (*timerAssembleGlobalMatrix);

			// "WORKSET CELL" loop: local cell ordinal is relative to numElems
			for (int cell = worksetBegin; cell < worksetEnd; ++cell) {

				// Compute cell ordinal relative to the current workset
				const int worksetCellOrdinal = cell - worksetBegin;

				// "CELL EQUATION" loop for the workset cell: cellRow is
				// relative to the cell DoF numbering.
				for (int cellRow = 0; cellRow < numFieldsG; ++cellRow) {
					int localRow  = elem_to_node (cell, cellRow);
					int globalRow = as<int> (global_node_ids[localRow]);
					//					ST sourceTermContribution = worksetSource (worksetCellOrdinal, cellRow);
					//					ArrayView<ST> sourceTermContributionAV =
					//							arrayView (&sourceTermContribution, 1);
					//
					//					SourceVector->sumIntoGlobalValue (globalRow, sourceTermContribution);

					// "CELL VARIABLE" loop for the workset cell: cellCol is
					// relative to the cell DoF numbering.
					for (int cellCol = 0; cellCol < numFieldsG; cellCol++){
						const int localCol  = elem_to_node(cell, cellCol);
						int globalCol = as<int> (global_node_ids[localCol]);
						ArrayView<int> globalColAV = arrayView<int> (&globalCol, 1);
						ST operatorMassMatrixContribution =
								worksetMassMatrix (worksetCellOrdinal, cellRow, cellCol);
						ST operatorStiffnessContribution =
								worksetStiffMatrix (worksetCellOrdinal, cellRow, cellCol);

//						if (((globalRow==0)||(globalCol==0))&&((globalRow==1)||(globalCol==1))) {
//							ArrayView<const ST> testRHS;
//							ArrayView<const int> constglobalColAV = arrayView<const int> (&globalCol, 1);
//							oRHS->getLocalRowView(globalRow,constglobalColAV,testRHS);
//							ArrayView<const ST> testLHS;
//							oLHS->getLocalRowView(globalRow,constglobalColAV,testLHS);
//							std::cout <<
//									" cum LHS before = "<< testLHS[globalCol] <<
//									" cum RHS before = "<< testRHS[globalCol] <<
//									std::endl;
//						}

						K->sumIntoGlobalValues (globalRow, globalColAV,
								arrayView<ST> (&operatorStiffnessContribution, 1));
						Mi->sumIntoGlobalValues (globalRow, globalColAV,
								arrayView<ST> (&operatorMassMatrixContribution, 1));

//						if (((globalRow==0)||(globalCol==0))&&((globalRow==1)||(globalCol==1))) {
//							ArrayView<const ST> testRHS;
//							ArrayView<const int> constglobalColAV = arrayView<const int> (&globalCol, 1);
//							oRHS->getLocalRowView(globalRow,constglobalColAV,testRHS);
//							ArrayView<const ST> testLHS;
//							oLHS->getLocalRowView(globalRow,constglobalColAV,testLHS);
//							std::cout << "r = " << globalRow << " c = " << globalCol<< "cell no = "<<worksetCellOrdinal<<
//									" M = " << worksetMassMatrix(worksetCellOrdinal,cellRow,cellCol) <<
//									" K = " << worksetStiffMatrix(worksetCellOrdinal,cellRow,cellCol) <<
//									" LHS = " << operatorMatrixContributionLHS <<
//									" RHS = " << operatorMatrixContributionRHS <<
//									" cum LHS after = "<< testLHS[globalCol] <<
//									" cum RHS after = "<< testRHS[globalCol] <<
//									std::endl;
//						}
					}// *** cell col loop ***
				}// *** cell row loop ***
			}// *** workset cell loop **
		} // *** stop timer ***
	}// *** workset loop ***

	K->fillComplete();
	Mi->fillComplete();
}

void Pde::compose_LHS_and_RHS() {
	  using Thyra::VectorSpaceBase;
	  using Thyra::VectorBase;
	  using Thyra::MultiVectorBase;
	  using Thyra::productVectorSpace;
	  using Thyra::DefaultProductVectorSpace;
	  using Thyra::LinearOpBase;
	  using Thyra::randomize;
	  using Thyra::identity;
	  using Thyra::diagonal;
	  using Thyra::multiply;
	  using Thyra::inverse;
	  using Thyra::add;
	  using Thyra::subtract;
	  using Thyra::zero;
	  using Thyra::scale;
	  using Thyra::adjoint;
	  using Thyra::block1x2;
	  using Thyra::block2x2;
	  using Thyra::block2x2;
	  using Thyra::transpose;
	  using Thyra::DefaultZeroLinearOp;
	  using Thyra::createVector;
	  using Thyra::createLinearOp;
	  using Thyra::createConstLinearOp;
	  typedef Thyra::LinearOpBase<ST> linearOp;
	  typedef Thyra::LinearOpWithSolveBase<ST> linearOpWithSolve;
	  typedef Thyra::LinearOpWithSolveFactoryBase<ST> linearOpWithSolveFactory;

	  Stratimikos::DefaultLinearSolverBuilder strategy;
	  strategy.paramsXmlFileName("params.xml");
	  strategy.readParameters(&std::cout);
	  lowsFactory = strategy.createLinearSolveStrategy("Belos");
	  /*
	   * wrap tpetra vectors and matricies
	   */

//	  X = createVector(U_Lambda);
//	  Y = createVector(U_Lambda_rhs);

	  RCP< VectorBase<ST> > U_w = createVector(U);
	  RCP< VectorBase<ST> > Lambda_w = createVector(Lambda);
	  RCP< VectorBase<ST> > U_rhs_w = createVector(U_rhs);
	  RCP< VectorBase<ST> > Lambda_rhs_w = createVector(Lambda_rhs);


	  Teuchos::Array<RCP<const VectorSpaceBase<ST> > > U_Lambda_space;
	  U_Lambda_space.push_back(U_w->space());
	  U_Lambda_space.push_back(Lambda_w->space());
	  RCP< DefaultProductVectorSpace<ST> > prodSpace = productVectorSpace<ST>(U_Lambda_space());

	  Teuchos::Array<RCP<VectorBase<ST> > > U_Lambda_vec,U_Lambda_rhs_vec;
	  U_Lambda_vec.push_back(U_w);
	  U_Lambda_vec.push_back(Lambda_w);
	  U_Lambda_rhs_vec.push_back(U_rhs_w);
	  U_Lambda_rhs_vec.push_back(Lambda_rhs_w);

	  X = Thyra::defaultProductVector<ST>(prodSpace,U_Lambda_vec());
	  Y = Thyra::defaultProductVector<ST>(prodSpace,U_Lambda_rhs_vec());

	  //std::cout << Teuchos::describe(*X,Teuchos::VERB_MEDIUM);
	  //RCP<const multivector_type> test = Thyra::TpetraOperatorVectorExtraction<ST,LO,GO,Node>::getConstTpetraMultiVector(X->getMultiVectorBlock(0));


	  RCP< const linearOp > Mi_w = createConstLinearOp<ST, LO, GO, Node>(Mi);
	  RCP< const linearOp > K_w = createConstLinearOp<ST, LO, GO, Node>(K);
	  RCP< const linearOp > Mb_w = createConstLinearOp<ST, LO, GO, Node>(Mb);
	  RCP< const linearOp > B_w = createConstLinearOp<ST, LO, GO, Node>(B);
	  RCP< const linearOp > omegaB = scale(omega,B_w);
	  RCP< const linearOp > omegaB_t = transpose(omegaB,"omegaB_t");

	  RCP<const linearOp>	zero11  = zero(Mb_w->range(),Mb_w->domain());
	  RCP<const linearOp>	zero01 = zero(omegaB_t->range(),omegaB_t->domain());
	  RCP<const linearOp>	zero10 = zero(B_w->range(),B_w->domain());

	  RCP<const linearOp> A = add(Mi_w,scale(omega*dt,K_w),"A");

	  RCP<const linearOp> LHSop = block2x2<ST>(
			  A, omegaB_t,
			  omegaB, zero11,
			  "LHSop");



	  RHS = block2x2(
			  add(Mi_w,scale((omega-1)*dt,K_w)), scale(omega-1,transpose(B_w,"B_t")),
			  scale(1-omega,B_w), zero11,
			  "RHS");

	  //std::cout << Teuchos::describe(*RHS,Teuchos::VERB_MEDIUM);


	  invA = inverse<ST>(*lowsFactory,A);

	  S = multiply(omegaB,invA,omegaB_t,"S");

	  RCP<const linearOp> LHS_prec = block2x2(A,zero01,
			  	  	  	   zero10,S,
			  	  	  	   "LHS_prec");

	  //std::cout << Teuchos::describe(*LHS_prec,Teuchos::VERB_MEDIUM);

	  LHS = lowsFactory->createOp();
	  //Thyra::initializeOp(*lowsFactory,LHSop,LHS.ptr());
	  //Thyra::initializeApproxPreconditionedOp(*lowsFactory,LHSop,LHS_prec,LHS.ptr());
	  Thyra::initializePreconditionedOp<ST>(*lowsFactory,LHSop,Thyra::unspecifiedPrec<ST>(LHS_prec),LHS.ptr());
	  //Thyra::initializePreconditionedOp<ST>(*lowsFactory,LHSop,Thyra::leftPrec<ST>(LHS_prec),LHS.ptr());


	  std::cout << Teuchos::describe(*LHS,Teuchos::VERB_MEDIUM);

}
