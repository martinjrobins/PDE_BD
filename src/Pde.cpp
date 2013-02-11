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

struct MyTrilinosData {
	RCP<sparse_matrix_type> A;
	RCP<vector_type> B, X;
};

Pde::Pde(const char* filename) {

	// Get the default communicator and Kokkos Node instance
	Teuchos::RCP<const Comm<int> > comm =
			Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
	Teuchos::RCP<Node> node = Tpetra::DefaultPlatform::getDefaultPlatform ().getNode ();


	std::string fstr(filename);
	if (fstr.substr(fstr.length-4,4) == ".xml") {
		Teuchos::ParameterList inputMeshList;
		LOG(2,"Reading mesh parameters from XML file \""<< fstr << "\"..." << std::endl);
		Teuchos::updateParametersFromXmlFile (fstr, inputMeshList);

		inputMeshList.print (std::out, 2, true, true);
		std::out << endl;

	} else {
		ERROR("unknown input filename to Pde class");
	}
#ifdef DEBUG
	const bool debug = true;
	const bool verbose = true;
#else
	const bool debug = false;
	const bool verbose = false;
#endif
	TrilinosRD::makeMatrixAndRightHandSide (data.A, data.B, data.X, comm, node, meshInput,
			std::out, std::err, verbose, debug);

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


