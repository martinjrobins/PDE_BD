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
#include "TrilinosRD.hpp"
#include "Log.h"

struct MyTrilinosData {
	RCP<sparse_matrix_type> A;
	RCP<vector_type> B, X;
};

Pde::Pde(const char* filename) {

	// Get the default communicator and Kokkos Node instance
	comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
	node = Tpetra::DefaultPlatform::getDefaultPlatform ().getNode ();


	std::string fstr(filename);
	if (fstr.substr(fstr.length-4,4) == ".xml") {
		Teuchos::ParameterList inputMeshList;
		LOG(2,"Reading mesh parameters from XML file \""<< fstr << "\"..." << std::endl);
		Teuchos::updateParametersFromXmlFile (fstr, inputMeshList);

		inputMeshList.print (std::out, 2, true, true);
		std::out << endl;
		setup_mesh();

	} else {
		ERROR("unknown input filename to Pde class");
	}

	make_LHS_and_RHS (data.A, data.B, data.X, comm, node, meshInput);

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

	// Get dimensions
	int numNodesPerElem = cellType.getNodeCount();
	int spaceDim = cellType.getDimension();
	int dim = 3;

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

	ASSERT(numElems > 0,"The number of elements in the mesh is zero.")

	long long numNodesGlobal;
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

	// Get boundary (side set) information
	long long * sideSetIds = new long long [numSideSets];
	long long numSidesInSet;
	long long numDFinSet;
	im_ex_get_side_set_ids_l(id,sideSetIds);
	for (int i=0; i < numSideSets; ++i) {
		im_ex_get_side_set_param_l (id,sideSetIds[i], &numSidesInSet, &numDFinSet);
		if (numSidesInSet > 0){
			long long * sideSetElemList = new long long [numSidesInSet];
			long long * sideSetSideList = new long long [numSidesInSet];
			im_ex_get_side_set_l (id, sideSetIds[i], sideSetElemList, sideSetSideList);
			for (int j = 0; j < numSidesInSet; ++j) {
				int sideNode0 = cellType.getNodeMap(2,sideSetSideList[j]-1,0);
				int sideNode1 = cellType.getNodeMap(2,sideSetSideList[j]-1,1);
				int sideNode2 = cellType.getNodeMap(2,sideSetSideList[j]-1,2);
				int sideNode3 = cellType.getNodeMap(2,sideSetSideList[j]-1,3);

				node_on_boundary(elem_to_node(sideSetElemList[j]-1,sideNode0))=1;
				node_on_boundary(elem_to_node(sideSetElemList[j]-1,sideNode1))=1;
				node_on_boundary(elem_to_node(sideSetElemList[j]-1,sideNode2))=1;
				node_on_boundary(elem_to_node(sideSetElemList[j]-1,sideNode3))=1;
			}
			delete [] sideSetElemList;
			delete [] sideSetSideList;
		}
	}
	delete [] sideSetIds;

	int numBCNodes = 0;
	for (int inode = 0; inode < numNodes; ++inode) {
		if (node_on_boundary(inode) && node_is_owned[inode]) {
			++numBCNodes;
		}
	}

	BCNodes.resize(numBCNodes);
	int indbc = 0;
	int iOwned = 0;
	for (int inode = 0; inode < numNodes; ++inode) {
		if (node_is_owned[inode]) {
			if (node_on_boundary (inode)) {
				BCNodes[indbc]=iOwned;
				++indbc;
			} // if node inode is on the boundary
			++iOwned;
		} // if node inode is owned by my process
	} // for each node inode that my process can see


	//
	// We're done with assembly, so we can delete the mesh.
	//

	Delete_Pamgen_Mesh ();
}

void Pde::create_cubature_and_basis() {
	using namespace Intrepid;
	/**********************************************************************************/
	/********************************* GET CUBATURE ***********************************/
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
}

void Pde::build_maps_and_create_matrices() {

	using Teuchos::Array;
	using Teuchos::TimeMonitor;
	using Teuchos::as;
	/**********************************************************************************/
	/********************* BUILD MAPS FOR GLOBAL SOLUTION *****************************/
	/**********************************************************************************/

	LOG(2,"Building Maps");
	int numFieldsG = HGradBasis->getCardinality();

	RCP<Teuchos::Time> timerBuildGlobalMaps =
			TimeMonitor::getNewTimer ("Build global Maps and Export");
	Array<int> ownedGIDs;
	const long long numNodes = global_node_ids.size();
	{
		TimeMonitor timerBuildGlobalMapsL (*timerBuildGlobalMaps);
		// Count owned nodes
		int ownedNodes = 0;
		for (int i = 0; i < numNodes; ++i) {
			if (node_is_owned[i]) {
				++ownedNodes;
			}
		}

		// Build a list of the OWNED global ids...
		// NTS: will need to switch back to long long
		ownedGIDs.resize(ownedNodes);
		int oidx = 0;
		for (int i = 0; i < numNodes; ++i) {
			if (node_is_owned[i]) {
				ownedGIDs[oidx] = as<int> (global_node_ids[i]);
				++oidx;
			}
		}
		globalMapG = rcp (new map_type (-1, ownedGIDs (), 0, comm, node));
	}

	/**********************************************************************************/
	/********************* BUILD MAPS FOR OVERLAPPED SOLUTION *************************/
	/**********************************************************************************/

	Array<GO> overlappedGIDs;
	{
		// Count owned nodes
		int overlappedNodes = numNodes;

		// Build a list of the OVERLAPPED global ids...
		overlappedGIDs.resize (overlappedNodes);
		for (int i = 0; i < numNodes; ++i) {
			overlappedGIDs[i] = as<int> (global_node_ids[i]);
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

				// "CELL EQUATION" loop for the workset cell: cellRow is
				// relative to the cell DoF numbering
				for (int cellRow = 0; cellRow < numFieldsG; cellRow++){

					int localRow  = elemToNode(cell, cellRow);
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
	F = rcp (new vector_type (globalMapG));
	X = rcp (new vector_type (globalMapG));

}

void Pde::integrate(const double dt) {
	std::cout << "integrating for "<<dt<<" seconds." << std::endl;
	bool converged = false;
	int numItersPerformed = 0;
	const MT tol = STM::squareroot (STM::eps ());
	const int maxNumIters = 100;
	TrilinosRD::solveWithBelos (converged, numItersPerformed, tol, maxNumIters,
			data.X, data.A, data.B, Teuchos::null, Teuchos::null);

	// Summarize timings
	Teuchos::RCP<ParameterList> reportParams = parameterList ("TimeMonitor::report");
	reportParams->set ("Report format", std::string ("YAML"));
	reportParams->set ("writeGlobalStats", true);
	Teuchos::TimeMonitor::report (*out, reportParams);
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
	Teuchos::OSTab tab (std::out);

	const int numFieldsG = HGradBasis->getCardinality();
	const int numCubPoints = cubature->getNumPoints();
	const int numElems = elem_to_node.dimension(0);
	const long long numNodes = global_node_ids.size();
	const int numNodesPerElem = elem_to_node.dimension(1);
	const int cubDim = cubature->getDimension();
	const int numBCNodes = BCNodes.size();

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

		// Copy coordinates into cell workset
		int cellCounter = 0;
		for (int cell = worksetBegin; cell < worksetEnd; ++cell) {
			for (int node = 0; node < numNodesPerElem; ++node) {
				cellWorkset(cellCounter, node, 0) = node_coord( elem_to_node(cell, node), 0);
				cellWorkset(cellCounter, node, 1) = node_coord( elem_to_node(cell, node), 1);
				cellWorkset(cellCounter, node, 2) = node_coord( elem_to_node(cell, node), 2);
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
		FieldContainer<ST> u_coeffs(worksetSize, numFieldsG);
		FieldContainer<ST> u_FE_val(worksetSize, numCubPoints);
		FieldContainer<ST> df_of_u(worksetSize, numCubPoints);
		FieldContainer<ST> df_of_u_times_basis(worksetSize, numFieldsG, numCubPoints);

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

		// Evaluate the source term at cubature points.
		evaluateSourceTerm (worksetSourceTerm, worksetCubPoints);

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
								+ theta*dt*worksetStiffMatrix (worksetCellOrdinal, cellRow, cellCol);
						ST operatorMatrixContributionRHS =
								worksetMassMatrix (worksetCellOrdinal, cellRow, cellCol)
								- (1.0-theta)*dt*worksetStiffMatrix (worksetCellOrdinal, cellRow, cellCol);

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

	LOG(2,"Exporting matrix and right-hand side from overlapped to owned Map");

	RCP<Teuchos::Time> timerAssembMultProc =
			TimeMonitor::getNewTimer ("Export from overlapped to owned");
	{
		TimeMonitor timerAssembMultProcL (*timerAssembMultProc);
		gl_StiffMatrix->setAllToScalar (STS::zero ());
		gl_StiffMatrix->doExport (*oLHS, *exporter, Tpetra::ADD);
		// If target of export has static graph, no need to do
		// setAllToScalar(0.0); export will clobber values.
		gl_StiffMatrix->fillComplete ();

		gl_SourceVector->putScalar (STS::zero ());
		gl_SourceVector->doExport (*SourceVector, *exporter, Tpetra::ADD);
	}

	//////////////////////////////////////////////////////////////////////////////
	// Adjust matrix for boundary conditions
	//////////////////////////////////////////////////////////////////////////////

	LOG(2,"Adjusting matrix and right-hand side for BCs");

	RCP<Teuchos::Time> timerAdjustMatrixBC =
			TimeMonitor::getNewTimer ("Adjust owned matrix and Source for BCs");
	{
		TimeMonitor timerAdjustMatrixBCL (*timerAdjustMatrixBC);

		// Apply owned stiffness matrix to v: rhs := A*v
		RCP<multivector_type> rhsDir =
				rcp (new multivector_type (globalMapG, true));
		gl_StiffMatrix->apply (*v.getConst (), *rhsDir);
		// Update right-hand side
		gl_SourceVector->update (as<ST> (-STS::one ()), *rhsDir, STS::one ());

		// Get a const view of the vector's local data.
		ArrayRCP<const ST> vArrayRCP = v->getData (0);

		// Adjust rhs due to Dirichlet boundary conditions.
		for (int inode = 0; inode < numNodes; ++inode) {
			if (node_is_owned[inode]) {
				if (node_on_boundary (inode)) {
					// Get global node ID
					const GO gni = as<GO> (global_node_ids[inode]);
					const LO lidT = globalMapG->getLocalElement (gni);
					ST v_valT = vArrayRCP[lidT];
					gl_SourceVector->replaceGlobalValue (gni, v_valT);
				}
			}
		}

		// Zero out rows and columns of stiffness matrix corresponding to
		// Dirichlet edges and add one to diagonal.  The following is the
		// Tpetra analog of Apply_OAZToMatrix().
		//
		// Reenable changes to the values and structure of the global
		// stiffness matrix.
		gl_StiffMatrix->resumeFill ();

		// Find the local column numbers to nuke
		RCP<const map_type> ColMap = gl_StiffMatrix->getColMap ();
		RCP<const map_type> globalMap =
				rcp (new map_type (gl_StiffMatrix->getGlobalNumCols (), 0, comm,
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
			const GO globalRow = gl_StiffMatrix->getRowMap ()->getGlobalElement (BCNodes[i]);
			const LO localCol = gl_StiffMatrix->getColMap ()->getLocalElement (globalRow);
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
		for (LO i = 0; i < as<int> (gl_StiffMatrix->getNodeNumRows ()); ++i) {
			NumEntries = gl_StiffMatrix->getNumEntriesInLocalRow (i);
			values.resize (NumEntries);
			indices.resize (NumEntries);
			gl_StiffMatrix->getLocalRowCopy (i, indices (), values (), NumEntries);
			for (int j = 0; j < as<int> (NumEntries); ++j) {
				if (myColsToZeroArrayRCP[indices[j]] == 1)
					values[j] = STS::zero ();
			}
			gl_StiffMatrix->replaceLocalValues (i, indices (), values ());
		} // for each (local) row of the global stiffness matrix

		// Zero the rows and add ones to diagonal.
		for (int i = 0; i < numBCNodes; ++i) {
			NumEntries = gl_StiffMatrix->getNumEntriesInLocalRow (BCNodes[i]);
			indices.resize (NumEntries);
			values.resize (NumEntries);
			gl_StiffMatrix->getLocalRowCopy (BCNodes[i], indices (), values (), NumEntries);
			const GO globalRow = gl_StiffMatrix->getRowMap ()->getGlobalElement (BCNodes[i]);
			const LO localCol = gl_StiffMatrix->getColMap ()->getLocalElement (globalRow);
			for (int j = 0; j < as<int> (NumEntries); ++j) {
				values[j] = STS::zero ();
				if (indices[j] == localCol) {
					values[j] = STS::one ();
				}
			} // for each entry in the current row
			gl_StiffMatrix->replaceLocalValues (BCNodes[i], indices (), values ());
		} // for each BC node
	}

	LOG(2,"Calling fillComplete() on owned-Map stiffness matrix");

	// We're done modifying the owned stiffness matrix.
	gl_StiffMatrix->fillComplete ();



	// Create vector to store approximate solution, and set initial guess.
	RCP<vector_type> gl_approxSolVector = rcp (new vector_type (globalMapG));
	gl_approxSolVector->putScalar (STS::zero ());

	// Store the output pointers.
	A = gl_StiffMatrix;
	B = gl_SourceVector;
	X = gl_approxSolVector;
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

void Pde::init(int argc, char *argv[]) {
	Teuchos::oblackholestream blackHole;
	Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackHole);
	my_rank = mpiSession.getRank();
	num_procs = mpiSession.getNProc();
}


