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



class Pde {
	typedef double ST;
	typedef int    LO;
	typedef int    GO;
	typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType  Node;


	typedef Tpetra::CrsMatrix<ST, LO, GO, Node>    sparse_matrix_type;
	typedef Tpetra::Operator<ST, LO, GO, Node>     operator_type;
	typedef Tpetra::MultiVector<ST, LO, GO, Node>  multivector_type;
	typedef Tpetra::Vector<ST, LO, GO, Node>       vector_type;

	typedef Tpetra::Map<LO, GO, Node>         map_type;
	typedef Tpetra::Export<LO, GO, Node>      export_type;
	typedef Tpetra::Import<LO, GO, Node>      import_type;
	typedef Tpetra::CrsGraph<LO, GO, Node>    sparse_graph_type;

	using Teuchos::RCP;
	using Teuchos::rcp;

public:
	Pde(const char* filename);
	~Pde() {
		delete [] node_is_owned;
		node_is_owned = NULL;
	}
	void integrate(const double dt);
	static void init(int argc, char *argv[]);
private:
	/*
	 * MPI
	 */
	int my_rank;
	int num_procs;
	RCP<const Teuchos::Comm<int> > comm;
	RCP<Node> node;

	/*
	 * Maps, exporters etc.
	 */
	RCP<sparse_graph_type> overlappedGraph;
	RCP<sparse_graph_type> ownedGraph;
	RCP<const map_type> globalMapG;
	RCP<const map_type> overlappedMapG;
	RCP<const export_type> exporter;

	/*
	 * Matricies
	 */
	RCP<sparse_matrix_type> LHS,RHS;
	RCP<vector_type> X,F;


	/*
	 * Mesh data
	 */
	const int spaceDim = 3;
	Intrepid::FieldContainer<int> elem_to_node;
	Intrepid::FieldContainer<ST> node_coord;
	Teuchos::Array<long long> global_node_ids;
	Teuchos::Array<int> BCNodes;
	Intrepid::FieldContainer<int> node_on_boundary;
	// nodeIsOwned must be a raw array, because std::vector<T> (and
	// therefore Teuchos::Array<T>) has a specialization for T = bool
	// that messes up the pointer type.
	// TODO: delete this when class is deleted....
	bool* node_is_owned;
	shards::CellTopology cellType;

	/*
	 * Basis
	 */
	const int cubDegree = 2;
	RCP<Intrepid::Cubature<ST> > cubature;
	Intrepid::FieldContainer<ST> cubPoints;
	Intrepid::FieldContainer<ST> cubWeights;
	RCP<Intrepid::Basis<ST, Intrepid::FieldContainer<ST> > >  HGradBasis;
	Intrepid::FieldContainer<ST> HGBValues;
	Intrepid::FieldContainer<ST> HGBGrads;

	void setup_pamgen_mesh(const std::string& meshInput);
	void create_cubature_and_basis();
	void build_maps_and_create_matrices();
	void make_LHS_and_RHS();


	template<typename Scalar>
	void materialTensor (Scalar material[][3],const Scalar& x,const Scalar& y,const Scalar& z);


	template<class ArrayOut, class ArrayIn>
	void evaluateMaterialTensor (ArrayOut& worksetMaterialValues,const ArrayIn& evaluationPoints);







};


#endif /* PDE_H_ */
