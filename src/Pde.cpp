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
#include "TrilinosSolve.h"
#include "Log.h"
#include "Constants.h"
#include "MyMpi.h"
#include <vtkPointData.h>
#include <set>


Pde::Pde(const ST dt, const ST dx):dt(dt),dirac_width(dx) {
	my_rank = Mpi::mpiSession->getRank();
	num_procs = Mpi::mpiSession->getNProc();

	// Get the default communicator and Kokkos Node instance
	comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
	node = Tpetra::DefaultPlatform::getDefaultPlatform ().getNode ();


	std::string meshInput;
	meshInput = makeMeshInput(1.0/dx, 1.0/dx, 1.0/dx);

	setup_pamgen_mesh(meshInput);


	create_cubature_and_basis();
	build_maps_and_create_matrices();
	create_vtk_grid();
	make_LHS_and_RHS();
}

void Pde::add_particle(const ST x, const ST y, const ST z) {
	typedef Teuchos::ScalarTraits<ST> STS;
	typedef STS::magnitudeType MT;
	typedef Teuchos::ScalarTraits<MT> STM;

	const int numNodes = node_coord.dimension(0);
	for (int i=0; i<numNodes; i++) {
		if (node_is_owned[i]) {
			const ST dx = node_coord(i,0) - x;
			const ST dy = node_coord(i,1) - y;
			const ST dz = node_coord(i,2) - z;
			if ((dx < dirac_width) && (dy < dirac_width) && (dz < dirac_width)) {
				const MT r = STM::squareroot(dx*dx + dy*dy + dz*dz);
				if (r < dirac_width) {
					X->sumIntoLocalValue(i,
							(1.0 - r/dirac_width)/(2.0*PI*PI*dirac_width)
					);
				} // if node within particle radius
			} // if node within the square
		} // if node is owned by this process
	} // loop through all nodes
}

struct fecomp{
	bool operator () ( topo_entity* x,  topo_entity*  y )const
	{
		if(x->sorted_local_node_ids < y->sorted_local_node_ids)return true;
		return false;
	}
};

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
	FieldContainer<int> refFaceToNode(numFacesPerElem,numNodesPerFace);
	for (int i=0; i<numFacesPerElem; i++){
		refFaceToNode(i,0)=cellType.getNodeMap(2, i, 0);
		refFaceToNode(i,1)=cellType.getNodeMap(2, i, 1);
		refFaceToNode(i,2)=cellType.getNodeMap(2, i, 2);
		refFaceToNode(i,3)=cellType.getNodeMap(2, i, 3);
	}

	// Build reference element face to edge map (Hardcoded for now)
	FieldContainer<int> refFaceToEdge(numFacesPerElem,numEdgesPerFace);
	refFaceToEdge(0,0)=0; refFaceToEdge(0,1)=9;
	refFaceToEdge(0,2)=4; refFaceToEdge(0,3)=8;
	refFaceToEdge(1,0)=1; refFaceToEdge(1,1)=10;
	refFaceToEdge(1,2)=5; refFaceToEdge(1,3)=9;
	refFaceToEdge(2,0)=2; refFaceToEdge(2,1)=11;
	refFaceToEdge(2,2)=6; refFaceToEdge(2,3)=10;
	refFaceToEdge(3,0)=8; refFaceToEdge(3,1)=7;
	refFaceToEdge(3,2)=11; refFaceToEdge(3,3)=3;
	refFaceToEdge(4,0)=3; refFaceToEdge(4,1)=2;
	refFaceToEdge(4,2)=1; refFaceToEdge(4,3)=0;
	refFaceToEdge(5,0)=4; refFaceToEdge(5,1)=5;
	refFaceToEdge(5,2)=6; refFaceToEdge(5,3)=7;

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



	 elemToFace.resize(numElems,numFacesPerElem);
	 //FieldContainer<int> elemToEdge(numElems,numEdgesPerElem);
	 std::set < topo_entity * , fecomp > edge_set;
	 std::set < topo_entity * , fecomp > face_set;
	 std::vector < topo_entity * > edge_vector;
	 std::vector < topo_entity * > face_vector;

	 // calculate edge and face ids
	 int elct = 0;
	 for(long long b = 0; b < numElemBlk; b++){
		 if(nodes_per_element[b] == 4){
		 }
		 else if (nodes_per_element[b] == 8){
			 //loop over all elements and push their edges onto a set if they are not there already
			 for(long long el = 0; el < elements[b]; el++){
				 std::set< topo_entity *, fecomp > ::iterator fit;
//				 for (int i=0; i < numEdgesPerElem; i++){
//					 topo_entity * teof = new topo_entity;
//					 for(int j = 0; j < numNodesPerEdge;j++){
//						 teof->add_node(elmt_node_linkage[b][el*numNodesPerElem + refEdgeToNode(i,j)],global_node_ids.begin().getRawPtr());
//					 }
//					 teof->sort();
//					 fit = edge_set.find(teof);
//					 if(fit == edge_set.end()){
//						 teof->local_id = edge_vector.size();
//						 edge_set.insert(teof);
//						 elemToEdge(elct,i)= edge_vector.size();
//						 edge_vector.push_back(teof);
//					 }
//					 else{
//						 elemToEdge(elct,i) = (*fit)->local_id;
//						 delete teof;
//					 }
//				 }
				 for (int i=0; i < numFacesPerElem; i++){
					 topo_entity * teof = new topo_entity;
					 for(int j = 0; j < numNodesPerFace;j++){
						 teof->add_node(elmt_node_linkage[b][el*numNodesPerElem + refFaceToNode(i,j)],global_node_ids.begin().getRawPtr());
					 }
					 teof->sort();
					 fit = face_set.find(teof);
					 if(fit == face_set.end()){
						 teof->local_id = face_vector.size();
						 face_set.insert(teof);
						 elemToFace(elct,i)= face_vector.size();
						 face_vector.push_back(teof);
					 }
					 else{
						 elemToFace(elct,i) = (*fit)->local_id;
						 delete teof;
					 }
				 }
				 elct ++;
			 }
		 }
	 }

	 int numFaces = face_vector.size();


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

	faceOnBoundary.resize(numFaces);

	// Get boundary (side set) information
	long long * sideSetIds = new long long [numSideSets];
	long long numSidesInSet;
	long long numDFinSet;
	int numBndyFaces=0;
	im_ex_get_side_set_ids_l(id,sideSetIds);
	for (int i=0; i < numSideSets; ++i) {
		im_ex_get_side_set_param_l (id,sideSetIds[i], &numSidesInSet, &numDFinSet);
		if (numSidesInSet > 0){
			long long * sideSetElemList = new long long [numSidesInSet];
			long long * sideSetSideList = new long long [numSidesInSet];
			im_ex_get_side_set_l (id, sideSetIds[i], sideSetElemList, sideSetSideList);
			for (int j = 0; j < numSidesInSet; ++j) {
				const int iface = sideSetSideList[j]-1;
				const int ielem = sideSetElemList[j]-1;

				int sideNode0 = cellType.getNodeMap(2,iface,0);
				int sideNode1 = cellType.getNodeMap(2,iface,1);
				int sideNode2 = cellType.getNodeMap(2,iface,2);
				int sideNode3 = cellType.getNodeMap(2,iface,3);

				faceOnBoundary(elemToFace(ielem,iface)) = 1;
				numBndyFaces++;

				node_on_boundary(elem_to_node(ielem,sideNode0)) = 1;
				node_on_boundary(elem_to_node(ielem,sideNode1)) = 1;
				node_on_boundary(elem_to_node(ielem,sideNode2)) = 1;
				node_on_boundary(elem_to_node(ielem,sideNode3)) = 1;
			}
			delete [] sideSetElemList;
			delete [] sideSetSideList;
		}
	}
	delete [] sideSetIds;


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

	// Evaluate basis values at cubature points
	faceBasis->getValues(faceValues, facePoints, OPERATOR_VALUE);

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
	Array<int> ownedGIDs;
	const long long numNodes = global_node_ids.size();
	int numBoundaryNodes = 0;
	{
		TimeMonitor timerBuildGlobalMapsL (*timerBuildGlobalMaps);
		// Count owned and boundary nodes
		int ownedNodes = 0;
		int ownedBoundaryNodes = 0;

		for (int i = 0; i < numNodes; ++i) {
			if (node_is_owned[i]) {
				++ownedNodes;
				if (node_on_boundary(i)) {
					++ownedBoundaryNodes;
				}
			}
			if (node_on_boundary(i)) {
				++numBoundaryNodes;
			}
		}



		ownedBCNodes.resize(ownedBoundaryNodes);
		// Build a list of the OWNED global ids...
		// NTS: will need to switch back to long long
		ownedGIDs.resize(ownedNodes+ownedBoundaryNodes);
		int oidx = 0;
		int obidx = 0;
		for (int i = 0; i < numNodes; ++i) {
			if (node_is_owned[i]) {
				ownedGIDs[oidx] = as<int> (global_node_ids[i]);
				++oidx;
				if (node_on_boundary(i)) {
					ownedGIDs[ownedNodes+obidx] = as<int> (global_node_ids[i]+numNodesGlobal);
					ownedBCNodes[obidx] = oidx;
					++obidx;
				}
			}
		}

		// extend list to include boundary condition nodes


		globalMapG = rcp (new map_type (-1, ownedGIDs (), 0, comm, node));
	}

	/**********************************************************************************/
	/********************* BUILD MAPS FOR OVERLAPPED SOLUTION *************************/
	/**********************************************************************************/

	Array<GO> overlappedGIDs;
	{
		// Count owned nodes
		int overlappedNodes = numNodes;
		BCNodes.resize(numBoundaryNodes);
		// Build a list of the OVERLAPPED global ids...
		overlappedGIDs.resize (overlappedNodes + numBoundaryNodes);
		int iBC = 0;
		for (int i = 0; i < numNodes; ++i) {
			overlappedGIDs[i] = as<int> (global_node_ids[i]);
			if (node_on_boundary(i)) {
				overlappedGIDs[i+numNodes] = as<int> (global_node_ids[i]+numNodesGlobal);
				BCNodes[iBC] = i;
				++iBC;
			}
		}

		//Generate overlapped Map for nodes.
		overlappedMapG = rcp (new map_type (-1, overlappedGIDs (), 0, comm, node));

		// Build Tpetra Export from overlapped to owned Map.
		exporter = rcp (new export_type (overlappedMapG, globalMapG));
	}

	/**********************************************************************************/
	/********************* BUILD GRAPH FOR OVERLAPPED SOLUTION ************************/
	/**********************************************************************************/

	LOG(2,"Building Graph");

	RCP<Teuchos::Time> timerBuildOverlapGraph =
			TimeMonitor::getNewTimer ("Build graphs for overlapped and owned solutions");
	const int numElems = elem_to_node.dimension(0);
	const int numFacesPerElem = cellType.getSideCount();
	{
		TimeMonitor timerBuildOverlapGraphL (*timerBuildOverlapGraph);

		// Construct Tpetra::CrsGraph objects.
		overlappedGraph = rcp (new sparse_graph_type (overlappedMapG, 0));
		ownedGraph = rcp (new sparse_graph_type (globalMapG, 0));

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


				for (int iface = 0; iface < numFacesPerElem; ++iface) {
					if (faceOnBoundary(elemToFace(cell,iface))) {
						for (int ipt = 0; ipt < numNodesPerFace; ++ipt) {

							for (int cellpt = 0; cellpt < numFieldsG; cellpt++) {
								int localCellpt  = elem_to_node(cell, cellpt);
								//globalRow for Tpetra Graph
								Tpetra::global_size_t globalCellT = as<Tpetra::global_size_t> (global_node_ids[localCellpt]);

								Tpetra::global_size_t globalBT = globalRowT + numNodesGlobal;
								int globalB = as<int> (globalBT);
								Teuchos::ArrayView<int> globalBAV = Teuchos::arrayView (&globalB, 1);
								overlappedGraph->insertGlobalIndices (globalRowT, globalBAV);
							}
						}
					}
				}
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
						overlappedGraph->insertGlobalIndices (globalRowT, globalColAV);
					}// *** cell col loop ***

					// add node -> boundary node connectivity
					if (node_on_boundary(cellRow)) {
						Tpetra::global_size_t globalBT = globalRowT + numNodesGlobal;
						int globalB = as<int> (globalBT);
						Teuchos::ArrayView<int> globalBAV = Teuchos::arrayView (&globalB, 1);
						overlappedGraph->insertGlobalIndices (globalRowT, globalBAV);
						// "CELL VARIABLE" loop for the workset cell: cellCol is
						// relative to the cell DoF numbering
						for (int cellCol = 0; cellCol < numFieldsG; ++cellCol) {
							int localCol  = elem_to_node (cell, cellCol);
							int globalCol = as<int> (global_node_ids[localCol]);
							//create ArrayView globalCol object for Tpetra
							Teuchos::ArrayView<int> globalColAV = Teuchos::arrayView (&globalCol, 1);

							//Update Tpetra overlap Graph
							overlappedGraph->insertGlobalIndices (globalBT, globalColAV);
						}// *** cell col loop ***
					}
				}// *** cell row loop ***
			}// *** workset cell loop **
		}// *** workset loop ***

		// Fill-complete overlapping distribution Graph.
		overlappedGraph->fillComplete ();

		// Export to owned distribution Graph, and fill-complete the latter.
		ownedGraph->doExport (*overlappedGraph, *exporter, Tpetra::INSERT);
		ownedGraph->fillComplete ();
	}

	LOG(2,"Constructing stiffness matrix and vectors");

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
	LHS = rcp (new sparse_matrix_type (ownedGraph.getConst ()));
	RHS = rcp (new sparse_matrix_type (ownedGraph.getConst ()));
	boundary_grad_op = rcp (new sparse_matrix_type (ownedGraph.getConst ()));
	//F = rcp (new vector_type (globalMapG));
	X = rcp (new vector_type (globalMapG));

	// initialise source term and concentration to zero
	//F->putScalar(0);
	X->putScalar(0);

}

void Pde::integrate(const ST requested_dt) {


	const int iterations = int(requested_dt/dt + 0.5);
	const double actual_dt = iterations*requested_dt;
	std::cout << "integrating for "<<actual_dt<<" seconds." << std::endl;
	for (int i = 0; i < iterations; ++i) {
		/*
		 * Solve FEM system
		 */
		solve();

		/*
		 * calculate gradient
		 */
		boundary_grad_op->apply(*X.getConst(),*X_grad);
	}
}


void Pde::make_LHS_and_RHS () {
	using namespace Intrepid;
	using Tpetra::global_size_t;
	using Teuchos::Array;
	using Teuchos::ArrayRCP;
	using Teuchos::ArrayView;
	using Teuchos::arrayView;
	using Teuchos::as;

	using Teuchos::TimeMonitor;
	typedef Teuchos::ArrayView<LO>::size_type size_type;
	typedef Teuchos::ScalarTraits<ST> STS;


	// Number of independent variables fixed at 3
	typedef Sacado::Fad::SFad<ST, 3>     Fad3;
	typedef Intrepid::FunctionSpaceTools IntrepidFSTools;
	typedef Intrepid::RealSpaceTools<ST> IntrepidRSTools;
	typedef Intrepid::CellTools<ST>      IntrepidCTools;

	LOG(2,"makeMatrixAndRightHandSide:");
	Teuchos::OSTab tab (std::cout);

	const int numFieldsG = HGradBasis->getCardinality();
	const int numCubPoints = cubature->getNumPoints();
	const int numElems = elem_to_node.dimension(0);
	const long long numNodes = global_node_ids.size();
	const int numNodesPerElem = elem_to_node.dimension(1);
	const int cubDim = cubature->getDimension();
	const int numBCNodes = ownedBCNodes.size();

	//
	// Overlapped distribution objects:
	//
	RCP<sparse_matrix_type> oLHS =
			rcp (new sparse_matrix_type (overlappedGraph.getConst ()));
	oLHS->setAllToScalar (STS::zero ());
	RCP<sparse_matrix_type> oRHS=
			rcp (new sparse_matrix_type (overlappedGraph.getConst ()));
	oRHS->setAllToScalar (STS::zero ());

	/**********************************************************************************/
	/******************** BOUNDARY CONDITIONS ***************************/
	/**********************************************************************************/

	const int faceDim       = faceCubature->getDimension ();
	const int numFacePoints = faceCubature->getNumPoints ();
	const int numFacesPerElem = cellType.getSideCount();

	FieldContainer<double> bndyFaceVal(numBndyFaces);
	FieldContainer<double> refFacePoints(numFacePoints,spaceDim);
	FieldContainer<double> bndyFacePoints(1,numFacePoints,spaceDim);
	FieldContainer<double> bndyFaceJacobians(1,numFacePoints,spaceDim,spaceDim);
	FieldContainer<double> faceNorm(1,numFacePoints,spaceDim);
	FieldContainer<double> uDotNormal(1,numFacePoints);
	FieldContainer<double> uFace(numFacePoints,spaceDim);
	FieldContainer<double> nodes(1, numNodesPerElem, spaceDim);

	int ibface=0;

	for (int ielem=0; ielem<numElems; ielem++) {
		for (int inode=0; inode<numNodesPerElem; inode++) {
			nodes(0,inode,0) = node_coord(elem_to_node(ielem,inode),0);
			nodes(0,inode,1) = node_coord(elem_to_node(ielem,inode),1);
			nodes(0,inode,2) = node_coord(elem_to_node(ielem,inode),2);
		}
		for (int iface=0; iface<numFacesPerElem; iface++){
			if(faceOnBoundary(elemToFace(ielem,iface))){

				// map evaluation points from reference face to reference cell
				IntrepidCTools::mapToReferenceSubcell(refFacePoints,
						facePoints,
						2, iface, cellType);

				// calculate Jacobian
				IntrepidCTools::setJacobian(bndyFaceJacobians, refFacePoints,
						nodes, cellType);

				// map evaluation points from reference cell to physical cell
				IntrepidCTools::mapToPhysicalFrame(bndyFacePoints,
						refFacePoints,
						nodes, cellType);

				// Compute face normals
				IntrepidCTools::getPhysicalFaceNormals(faceNorm,
						bndyFaceJacobians,
						iface, cellType);

				// evaluate exact solution and dot with normal
				for(int nPt = 0; nPt < numFacePoints; nPt++){

					double x = bndyFacePoints(0, nPt, 0);
					double y = bndyFacePoints(0, nPt, 1);
					double z = bndyFacePoints(0, nPt, 2);

					evalu(uFace(nPt,0), uFace(nPt,1), uFace(nPt,2), x, y, z);
					uDotNormal(0,nPt)=(uFace(nPt,0)*faceNorm(0,nPt,0)+uFace(nPt,1)*faceNorm(0,nPt,1)+uFace(nPt,2)*faceNorm(0,nPt,2));
				}

				// integrate
				for(int nPt = 0; nPt < numFacePoints; nPt++){
					bndyFaceVal(ibface)=bndyFaceVal(ibface)+uDotNormal(0,nPt)*paramFaceWeights(nPt);
				}
				ibface++;
			} // end if face on boundary

		} // end loop over element faces

	} // end loop over elements



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
		FieldContainer<ST> cellWorkset (worksetSize, numNodesPerElem, spaceDim);

		// array to contain boundary normals (=0 if not on boundary)
		FieldContainer<ST> boundary_normals(worksetSize, spaceDim);

		// Copy coordinates into cell workset
		int cellCounter = 0;
		for (int cell = worksetBegin; cell < worksetEnd; ++cell) {
			for (int node = 0; node < numNodesPerElem; ++node) {
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
						ST operatorMatrixContributionLHS =
								worksetMassMatrix (worksetCellOrdinal, cellRow, cellCol)
								+ omega*dt*worksetStiffMatrix (worksetCellOrdinal, cellRow, cellCol);
						ST operatorMatrixContributionRHS =
								worksetMassMatrix (worksetCellOrdinal, cellRow, cellCol)
								- (1.0-omega)*dt*worksetStiffMatrix (worksetCellOrdinal, cellRow, cellCol);

						oLHS->sumIntoGlobalValues (globalRow, globalColAV,
								arrayView<ST> (&operatorMatrixContributionLHS, 1));
						oRHS->sumIntoGlobalValues (globalRow, globalColAV,
								arrayView<ST> (&operatorMatrixContributionRHS, 1));
					}// *** cell col loop ***
				}// *** cell row loop ***
			}// *** workset cell loop **
		} // *** stop timer ***
	}// *** workset loop ***

	//////////////////////////////////////////////////////////////////////////////
	// Export sparse matrix and right-hand side from overlapping row Map
	// to owned (nonoverlapping) row Map.
	//////////////////////////////////////////////////////////////////////////////

	LOG(2,"Exporting RHS and LHS from overlapped to owned Map");

	RCP<Teuchos::Time> timerAssembMultProc =
			TimeMonitor::getNewTimer ("Export from overlapped to owned");
	{
		TimeMonitor timerAssembMultProcL (*timerAssembMultProc);
		LHS->setAllToScalar (STS::zero ());
		LHS->doExport (*oLHS, *exporter, Tpetra::ADD);
		// If target of export has static graph, no need to do
		// setAllToScalar(0.0); export will clobber values.
		LHS->fillComplete ();

		RHS->setAllToScalar (STS::zero ());
		RHS->doExport (*oRHS, *exporter, Tpetra::ADD);
		// If target of export has static graph, no need to do
		// setAllToScalar(0.0); export will clobber values.
		RHS->fillComplete ();
	}

	//////////////////////////////////////////////////////////////////////////////
	// Adjust matrix for boundary conditions
	//////////////////////////////////////////////////////////////////////////////

//	LOG(2,"Adjusting for BCs");
//
//	RCP<Teuchos::Time> timerAdjustMatrixBC =
//			TimeMonitor::getNewTimer ("Adjust owned matrix for BCs");
//	{
//		TimeMonitor timerAdjustMatrixBCL (*timerAdjustMatrixBC);
//		// Zero out rows and columns of LHS & RHS matrix corresponding to
//		zero_out_rows_and_columns(LHS);
//		zero_out_rows_and_columns(RHS);
//	}

}

void Pde::zero_out_rows_and_columns(RCP<sparse_matrix_type> matrix) {
	using Teuchos::Array;
	using Teuchos::ArrayRCP;
	using Teuchos::ArrayView;
	using Teuchos::arrayView;
	using Teuchos::as;
	typedef Teuchos::ScalarTraits<ST> STS;

	const int numBCNodes = ownedBCNodes.size();

	// Zero out rows and columns of LHS & RHS matrix corresponding to
	// Dirichlet edges and add one to diagonal.  The following is the
	// Tpetra analog of Apply_OAZToMatrix().
	//
	// Reenable changes to the values and structure of the global
	// stiffness matrix.
	matrix->resumeFill ();

	// Find the local column numbers to nuke
	RCP<const map_type> ColMap = matrix->getColMap ();
	RCP<const map_type> globalMap =
			rcp (new map_type (matrix->getGlobalNumCols (), 0, comm,
					Tpetra::GloballyDistributed, node));

	// Create the exporter from this process' column Map to the global
	// 1-1 column map. (???)
	RCP<const export_type> bdyExporter =
			rcp (new export_type (ColMap, globalMap));
	// Create a vector of global column indices to which we will export
	RCP<Tpetra::Vector<int, LO, GO, Node> > globColsToZeroT =
			rcp (new Tpetra::Vector<int, LO, GO, Node> (globalMap));
	// Create a vector of local column indices from which we will export
	RCP<Tpetra::Vector<int, LO, GO, Node> > myColsToZeroT =
			rcp (new Tpetra::Vector<int, LO, GO, Node> (ColMap));
	myColsToZeroT->putScalar (0);

	// Flag (set to 1) all local columns corresponding to the local
	// rows specified.
	for (int i = 0; i < numBCNodes; ++i) {
		const GO globalRow = matrix->getRowMap ()->getGlobalElement (ownedBCNodes[i]);
		const LO localCol = matrix->getColMap ()->getLocalElement (globalRow);
		// Tpetra::Vector<int, ...> works just like
		// Tpetra::Vector<double, ...>.  Epetra has a separate
		// Epetra_IntVector class for ints.
		myColsToZeroT->replaceLocalValue (localCol, 1);
	}

	// Export to the global column map.
	globColsToZeroT->doExport (*myColsToZeroT, *bdyExporter, Tpetra::ADD);
	// Import from the global column map to the local column map.
	myColsToZeroT->doImport (*globColsToZeroT, *bdyExporter, Tpetra::INSERT);

	Array<ST> values;
	Array<int> indices;
	ArrayRCP<const int> myColsToZeroArrayRCP = myColsToZeroT->getData(0);
	size_t NumEntries = 0;

	// Zero the columns corresponding to Dirichlet BCs.
	for (LO i = 0; i < as<int> (matrix->getNodeNumRows ()); ++i) {
		NumEntries = matrix->getNumEntriesInLocalRow (i);
		values.resize (NumEntries);
		indices.resize (NumEntries);
		matrix->getLocalRowCopy (i, indices (), values (), NumEntries);
		for (int j = 0; j < as<int> (NumEntries); ++j) {
			if (myColsToZeroArrayRCP[indices[j]] == 1)
				values[j] = STS::zero ();
		}
		matrix->replaceLocalValues (i, indices (), values ());
	} // for each (local) row of the global stiffness matrix

	// Zero the rows and add ones to diagonal.
	for (int i = 0; i < numBCNodes; ++i) {
		NumEntries = matrix->getNumEntriesInLocalRow (ownedBCNodes[i]);
		indices.resize (NumEntries);
		values.resize (NumEntries);
		matrix->getLocalRowCopy (ownedBCNodes[i], indices (), values (), NumEntries);
		const GO globalRow = matrix->getRowMap ()->getGlobalElement (ownedBCNodes[i]);
		const LO localCol = matrix->getColMap ()->getLocalElement (globalRow);
		for (int j = 0; j < as<int> (NumEntries); ++j) {
			values[j] = STS::zero ();
			if (indices[j] == localCol) {
				values[j] = STS::one ();
			}
		} // for each entry in the current row
		matrix->replaceLocalValues (ownedBCNodes[i], indices (), values ());
	} // for each BC node

	// We're done modifying the owned stiffness matrix.
	matrix->fillComplete ();

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


	bool converged = false;
	int numItersPerformed = 0;
	const MT tol = STM::squareroot (STM::eps ());
	const int maxNumIters = 100;
	RCP<vector_type> rhsDir =
			rcp (new vector_type (globalMapG, true));
	RHS->apply(*X.getConst(),*rhsDir);
	TrilinosRD::solveWithBelos<ST, multivector_type, operator_type>(
			converged, numItersPerformed, tol, maxNumIters,
			X, LHS,
			rhsDir, Teuchos::null, Teuchos::null
			);

	// Summarize timings
//	Teuchos::RCP<Teuchos::ParameterList> reportParams = parameterList ("TimeMonitor::report");
//	reportParams->set ("Report format", std::string ("YAML"));
//	reportParams->set ("writeGlobalStats", true);
//	Teuchos::TimeMonitor::report(std::cout, reportParams);
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
     << "\t\tsideset, ilo, 1" << endl
     << "\t\tsideset, jlo, 2" << endl
     << "\t\tsideset, klo, 3" << endl
     << "\t\tsideset, ihi, 4" << endl
     << "\t\tsideset, jhi, 5" << endl
     << "\t\tsideset, khi, 6" << endl
     << "\tend" << endl
     << "end";
  return os.str ();
}

vtkUnstructuredGrid* Pde::get_grid() {
	return vtk_grid;
}

void Pde::create_vtk_grid() {
	/*
	 * setup points
	 */
	vtkSmartPointer<vtkPoints> newPts = vtkSmartPointer<vtkPoints>::New();
	const int num_points = node_coord.dimension(0);
	for (int i = 0; i < num_points; i++) {
		newPts->InsertNextPoint(node_coord(i,0),node_coord(i,1),node_coord(i,2));
	}

	/*
	 * setup scalar data
	 */
	vtkSmartPointer<vtkDoubleArray> newScalars = vtkSmartPointer<vtkDoubleArray>::New();
	const int num_local_entries = X->getLocalLength();
	ASSERT(num_local_entries == num_points, "size in X vector not same as number of points");
	newScalars->SetArray(X->getDataNonConst(0).getRawPtr(),num_local_entries,1);
	newScalars->SetName("Concentration");

	/*
	 * setup scalar grad data
	 */
	vtkSmartPointer<vtkDoubleArray> newScalars = vtkSmartPointer<vtkDoubleArray>::New();
	newScalars->SetArray(X_grad->getDataNonConst(0).getRawPtr(),num_local_entries,1);
	newScalars->SetName("Concentration Gradient");

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
}


