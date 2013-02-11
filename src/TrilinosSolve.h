/*
 * TrilinosSolve.h
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

#ifndef TRILINOSSOLVE_H_
#define TRILINOSSOLVE_H_


#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"


namespace TrilinosRD {


template<class ST, class MV, class OP>
void
solveWithBelos (bool& converged,
                int& numItersPerformed,
                const typename Teuchos::ScalarTraits<ST>::magnitudeType& tol,
                const int maxNumIters,
                const Teuchos::RCP<MV>& X,
                const Teuchos::RCP<const OP>& A,
                const Teuchos::RCP<const MV>& B,
                const Teuchos::RCP<const OP>& M_left,
                const Teuchos::RCP<const OP>& M_right)
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  typedef Belos::LinearProblem<ST, MV, OP > problem_type;
  typedef Belos::PseudoBlockCGSolMgr<ST, MV, OP> solver_type;
  typedef Belos::MultiVecTraits<ST, MV> MVT;

  // Set these in advance, so that if the Belos solver throws an
  // exception for some reason, these will have sensible values.
  converged = false;
  numItersPerformed = 0;

  TEUCHOS_TEST_FOR_EXCEPTION(A.is_null () || X.is_null () || B.is_null (),
    std::invalid_argument, "solveWithBelos: The A, X, and B arguments must all "
    "be nonnull.");
  const int numColsB = MVT::GetNumberVecs (*B);
  const int numColsX = MVT::GetNumberVecs (*X);
  TEUCHOS_TEST_FOR_EXCEPTION(numColsB != numColsX, std::invalid_argument,
    "solveWithBelos: X and B must have the same number of columns.  X has "
    << numColsX << " columns, but B has " << numColsB << " columns.");

  RCP<ParameterList> belosParams = parameterList ();
  belosParams->set ("Block Size", numColsB);
  belosParams->set ("Maximum Iterations", maxNumIters);
  belosParams->set ("Convergence Tolerance", tol);

  RCP<problem_type> problem = rcp (new problem_type (A, X, B));
  if (! M_left.is_null ()) {
    problem->setLeftPrec (M_left);
  }
  if (! M_right.is_null ()) {
    problem->setRightPrec (M_right);
  }
  const bool set = problem->setProblem ();
  TEUCHOS_TEST_FOR_EXCEPTION(! set, std::runtime_error, "solveWithBelos: The "
    "Belos::LinearProblem's setProblem() method returned false.  This probably "
    "indicates that there is something wrong with A, X, or B.");

  solver_type solver (problem, belosParams);
  Belos::ReturnType result = solver.solve ();

  converged = (result == Belos::Converged);
  numItersPerformed = solver.getNumIters ();
}

} // namespace TrilinosRD


#endif /* TRILINOSSOLVE_H_ */
