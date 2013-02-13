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

public:
	Pde(const char* filename);
	void integrate(const double dt);
	//static void init();
private:
	RCP<sparse_matrix_type> A,B;
	RCP<vector_type> X;
	Teuchos::RCP<const Comm<int> > comm;
	Teuchos::RCP<Node> node;

	/** \brief  User-defined material tensor.

	    \param  material    [out]   3 x 3 material tensor evaluated at (x,y,z)
	    \param  x           [in]    x-coordinate of the evaluation point
	    \param  y           [in]    y-coordinate of the evaluation point
	    \param  z           [in]    z-coordinate of the evaluation point

	    \warning Symmetric and positive definite tensor is required for every (x,y,z).
	*/
	template<typename Scalar>
	void materialTensor (Scalar material[][3],
	                const Scalar& x,
	                const Scalar& y,
	                const Scalar& z);

	/** \brief Compute the material tensor at array of points in physical space.

	    \param worksetMaterialValues      [out]     Rank-2, 3 or 4 array with dimensions (C,P), (C,P,D) or (C,P,D,D)
	                                                with the values of the material tensor
	    \param evaluationPoints           [in]      Rank-3 (C,P,D) array with the evaluation points in physical frame
	*/
	template<class ArrayOut, class ArrayIn>
	void evaluateMaterialTensor (ArrayOut& worksetMaterialValues,
	                        const ArrayIn& evaluationPoints);
	/** \brief Computes source term: f

	    \param  x          [in]    x-coordinate of the evaluation point
	    \param  y          [in]    y-coordinate of the evaluation point
	    \param  z          [in]    z-coordinate of the evaluation point

	    \return Source term corresponding to the user-defined exact solution evaluated at (x,y,z)
	 */
	template<typename Scalar>
	const Scalar sourceTerm (Scalar& x, Scalar& y, Scalar& z);

	void
	makeMatrixAndRightHandSide (Teuchos::RCP<sparse_matrix_type>& A,
	                            Teuchos::RCP<vector_type>& B,
	                            Teuchos::RCP<vector_type>& X_exact,
	                            Teuchos::RCP<vector_type>& X,
	                            const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
	                            const Teuchos::RCP<Node>& node,
	                            const std::string& meshInput);

	//! Just like above, but with multivector_type output arguments.
	void
	makeMatrixAndRightHandSide (Teuchos::RCP<sparse_matrix_type>& A,
	                            Teuchos::RCP<multivector_type>& B,
	                            Teuchos::RCP<multivector_type>& X_exact,
	                            Teuchos::RCP<multivector_type>& X,
	                            const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
	                            const Teuchos::RCP<Node>& node,
	                            const std::string& meshInput);

};


#endif /* PDE_H_ */
