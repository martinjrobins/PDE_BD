/* 
 * Pde.h
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

#ifndef PDE_H_
#define PDE_H_

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Vector.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"


// Intrepid includes
#include <Intrepid_FunctionSpaceTools.hpp>
#include <Intrepid_CellTools.hpp>
#include <Intrepid_ArrayTools.hpp>
#include <Intrepid_HGRAD_HEX_C1_FEM.hpp>
#include <Intrepid_RealSpaceTools.hpp>
#include <Intrepid_DefaultCubatureFactory.hpp>
#include <Intrepid_Utils.hpp>

// Teuchos includes
//#include <Teuchos_TimeMonitor.hpp>

// Shards includes
#include <Shards_CellTopology.hpp>

// Pamgen includes
#include <create_inline_mesh.h>
#include <im_exodusII_l.h>
#include <im_ne_nemesisI_l.h>
#include <pamgen_extras.h>

// Sacado includes
#include <Sacado.hpp>

// Belos includes
#include <BelosTpetraAdapter.hpp>
#include "TrilinosSolve.h"

// vtk includes
#include "vtkUnstructuredGrid.h"
#include "vtkSmartPointer.h"
#include "vtkDoubleArray.h"
#include "vtkHexahedron.h"

// STK includes
#include "stk_util/parallel/Parallel.hpp"
#include "stk_mesh/base/FieldData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Comm.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/GetBuckets.hpp"
#include "stk_mesh/fem/CreateAdjacentEntities.hpp"

// Bring in all of the operator/vector ANA client support software
#include "Thyra_OperatorVectorClientSupport.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_LinearSolverBuilderBase.hpp"
#include "Thyra_PreconditionerFactoryHelpers.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"


using Teuchos::RCP;
using Teuchos::rcp;

class Pde {

public:

	typedef double ST;
	typedef int    LO;
	typedef int    GO;
	//typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType  Node;
	typedef Kokkos::TPINode Node;
	typedef Tpetra::CrsMatrix<ST, LO, GO, Node>    sparse_matrix_type;
	typedef Tpetra::Operator<ST, LO, GO, Node>     operator_type;
	typedef Tpetra::MultiVector<ST, LO, GO, Node>  multivector_type;

	typedef Tpetra::Vector<ST, LO, GO, Node>       vector_type;

	typedef Tpetra::Map<LO, GO, Node>         map_type;
	typedef Tpetra::Export<LO, GO, Node>      export_type;
	typedef Tpetra::Import<LO, GO, Node>      import_type;
	typedef Tpetra::CrsGraph<LO, GO, Node>    sparse_graph_type;


	Pde(const ST dt, const ST dx, const int test_no=1, const int numThreads=1);
	~Pde() {
		delete [] node_is_owned;
		node_is_owned = NULL;
		// Get a summary from the time monitor.
		Teuchos::TimeMonitor::summarize();
	}
	void integrate(const ST dt);
	void add_particle(const ST x, const ST y, const ST z);
	void add_particles(std::vector<int>& points_added, const std::vector<double>& x,
			 const std::vector<double>& y, const std::vector<double>& z);

	vtkUnstructuredGrid* get_grid();
	vtkUnstructuredGrid* get_boundary();
	RCP<vector_type> get_boundary_node_values();
	RCP<vector_type> get_boundary_node_areas();
	RCP<multivector_type> get_boundary_node_positions();
	int get_total_number_of_particles();
	void rescale(double s);

	constexpr static double ri = 0.3;
	constexpr static double ro = 0.8;
private:

	/*
	 * MPI
	 */
	int my_rank;
	int num_procs;

	RCP<const Teuchos::Comm<int> > comm;
	RCP<Node> node;


	/*
	 * Matricies
	 */
	RCP<sparse_matrix_type> K, Mi, B, Mb;
	RCP<vector_type> U,Lambda,U_rhs,Lambda_rhs,U_Lambda,U_Lambda_rhs;

	RCP<vector_type> interior_node_volumes;
	RCP<vector_type> boundary_node_areas;
	RCP<multivector_type> boundary_node_positions;

	/*
	 * composed ops
	 */
	RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory;
	RCP<Thyra::LinearOpWithSolveBase<ST> > LHS;
	RCP<const Thyra::LinearOpBase<ST> > RHS;
	RCP<Thyra::VectorBase<ST> > X;
	RCP<Thyra::VectorBase<ST> > Y;


	/*
	 * Mesh data
	 */
	ST dirac_width;
	const static int spaceDim = 3;

	long long numNodesGlobal;
	int numNodesPerFace;
	Intrepid::FieldContainer<int> elem_to_node;
	Intrepid::FieldContainer<ST> node_coord;
	Teuchos::Array<long long> global_node_ids;

	Teuchos::Array<int> BCNodes;
	Intrepid::FieldContainer<int> node_on_boundary;
	Intrepid::FieldContainer<int> node_on_neumann;
	Intrepid::FieldContainer<int> node_on_boundary_id;

	Intrepid::FieldContainer<int> boundary_face_to_elem;
	Intrepid::FieldContainer<int> boundary_face_to_ordinal;
	bool* node_is_owned;
	shards::CellTopology cellType;
	shards::CellTopology faceType;
	Intrepid::FieldContainer<int> refFaceToNode;


	/*
	 * Basis
	 */
	const static int cubDegree = 2;
	RCP<Intrepid::Cubature<ST> > cubature;
	Intrepid::FieldContainer<ST> ref_points;
	Intrepid::FieldContainer<ST> cubWeights;
	Intrepid::FieldContainer<ST> cubPoints;


	RCP<Intrepid::Basis<ST, Intrepid::FieldContainer<ST> > >  HGradBasis;
	Intrepid::FieldContainer<ST> HGBValues;
	Intrepid::FieldContainer<ST> HGBFaceValues;
	Intrepid::FieldContainer<ST> HGBGrads;

	RCP<Intrepid::Basis<ST, Intrepid::FieldContainer<ST> > >  faceBasis;
	RCP<Intrepid::Cubature<ST> > faceCubature;
	Intrepid::FieldContainer<ST> faceValues;
	Intrepid::FieldContainer<ST> facePoints;
	Intrepid::FieldContainer<ST> faceWeights;

	/*
	 * Timestepping
	 */
	ST dt;
	static const ST omega;

	/*
	 * vtk unstructured grid
	 */
	vtkSmartPointer<vtkUnstructuredGrid> vtk_grid,vtk_boundary;


	void setup_pamgen_mesh(const std::string& meshInput);
	void create_cubature_and_basis();
	void build_maps_and_create_matrices();
	void fillMatricies();
	void compose_LHS_and_RHS();
	void calculate_volumes_and_areas();
	void boundary_integrals();
	void volume_integrals();
	void solve();
	std::string makeMeshInput (const int nx, const int ny, const int nz);
	std::string makeMeshInputFullDomain (const int nx, const int ny, const int nz);
	std::string makeMeshInputSphere (const int nr, const int ntheta);
	std::string makeMeshInputRadialTrisection (const int nr, const int ntheta, const int nz);
	std::string makeMeshInputCylinder (const int nr, const int ntheta, const int nz);

	void create_vtk_grid();
	void create_stk_grid();


	template<typename Scalar>
	void materialTensor (Scalar material[][3],const Scalar& x,const Scalar& y,const Scalar& z);


	template<class ArrayOut, class ArrayIn>
	void evaluateMaterialTensor (ArrayOut& worksetMaterialValues,const ArrayIn& evaluationPoints);


};


#endif /* PDE_H_ */
