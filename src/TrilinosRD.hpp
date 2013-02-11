/*
 * TrilinosHeader.hpp
 * 
 * Copyright 2013 Martin Robinson
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
 *  Created on: 11 Feb 2013
 *      Author: robinsonm
 */

#ifndef TRILINOSHEADER_HPP_
#define TRILINOSHEADER_HPP_

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


typedef double ST;
typedef int    LO;
typedef int    GO;
typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType  Node;


typedef Tpetra::CrsMatrix<ST, LO, GO, Node>    sparse_matrix_type;
typedef Tpetra::Operator<ST, LO, GO, Node>     operator_type;
typedef Tpetra::MultiVector<ST, LO, GO, Node>  multivector_type;
typedef Tpetra::Vector<ST, LO, GO, Node>       vector_type;


namespace TrilinosRD {


void
makeMatrixAndRightHandSide (Teuchos::RCP<sparse_matrix_type>& A,
                            Teuchos::RCP<vector_type>& B,
                            Teuchos::RCP<vector_type>& X_exact,
                            Teuchos::RCP<vector_type>& X,
                            const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                            const Teuchos::RCP<Node>& node,
                            const std::string& meshInput,
                            const Teuchos::RCP<Teuchos::FancyOStream>& out,
                            const Teuchos::RCP<Teuchos::FancyOStream>& err,
                            const bool verbose = false,
                            const bool debug = false);

//! Just like above, but with multivector_type output arguments.
void
makeMatrixAndRightHandSide (Teuchos::RCP<sparse_matrix_type>& A,
                            Teuchos::RCP<multivector_type>& B,
                            Teuchos::RCP<multivector_type>& X_exact,
                            Teuchos::RCP<multivector_type>& X,
                            const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                            const Teuchos::RCP<Node>& node,
                            const std::string& meshInput,
                            const Teuchos::RCP<Teuchos::FancyOStream>& out,
                            const Teuchos::RCP<Teuchos::FancyOStream>& err,
                            const bool verbose = false,
                            const bool debug = false);

//! Return \f$\|B - A X_{\text{exact}}\|_2\f$, \f$\|B\|\f$, and \f$\|A\|_F\f$.
std::vector<Teuchos::ScalarTraits<ST>::magnitudeType>
exactResidualNorm (const Teuchos::RCP<const sparse_matrix_type>& A,
                   const Teuchos::RCP<const vector_type>& B,
                   const Teuchos::RCP<const vector_type>& X_exact);


void
solveWithBelos (bool& converged,
                int& numItersPerformed,
                const Teuchos::ScalarTraits<ST>::magnitudeType& tol,
                const int maxNumIters,
                const Teuchos::RCP<multivector_type>& X,
                const Teuchos::RCP<const sparse_matrix_type>& A,
                const Teuchos::RCP<const multivector_type>& B,
                const Teuchos::RCP<const operator_type>& M_left=Teuchos::null,
                const Teuchos::RCP<const operator_type>& M_right=Teuchos::null);



}  // namespace TrilinosRD


#endif /* TRILINOSHEADER_HPP_ */
