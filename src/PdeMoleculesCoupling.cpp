/*
 * PdeMoleculesCoupling.cpp
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
 *  Created on: 28 Feb 2013
 *      Author: robinsonm
 */

#include "PdeMoleculesCoupling.h"
#include <MatrixMarket_Tpetra.hpp>
//#include <math.h>

#include <Tpetra_RTI.hpp>

int find_first(const double to_find, const std::vector<double>& to_find_in) {
	if (to_find_in[0] >= to_find) return 0;
	for (int i = 1; i < to_find_in.size(); ++i) {
		if (to_find_in[i] >= to_find) {
			if (to_find_in[i]-to_find < to_find-to_find_in[i-1]) {
				return i;
			} else {
				return i-1;
			}
		}
	}
}


void PdeMoleculesCoupling::generate_new_molecules(MoleculesSimple& mols,
		Pde& pde, const double dt, const double dA, const double D) {
	RCP<Pde::vector_type> boundary_node_values = pde.get_boundary_node_values();
	RCP<Pde::multivector_type> boundary_node_positions = pde.get_boundary_node_positions();

	std::vector<double> bnv_cumsum(boundary_node_values->getLocalLength());
	boundary_node_values->get1dCopy(Teuchos::ArrayView<double>(bnv_cumsum));
	for (int i = 1; i < bnv_cumsum.size(); ++i) {
		bnv_cumsum[i] += std::max(bnv_cumsum[i-1],0.0);
	}

	const double sum_values = *(bnv_cumsum.end()-1);
	const double Ld = dA*sum_values/dt;

	boost::variate_generator<base_generator_type&, boost::uniform_real<> >
				U(R,boost::uniform_real<>(0,1));
	boost::variate_generator<base_generator_type&, boost::normal_distribution<> >
				N(R,boost::normal_distribution<>(0,1));

//	std::cout << "x vector = " << std::endl;
//	boundary_node_positions->getVector(0)->print(std::cout);
//	Tpetra::MatrixMarket::Writer<Pde::vector_type>::writeDense (std::cout, boundary_node_positions->getVector(0));

	double tau = -log(U())/Ld;
	while (tau < dt) {
		const double r = U()*sum_values;
		const int r_find = find_first(r,bnv_cumsum);
		const double step_length = sqrt(2.0*D*(dt-tau));
		const double x = boundary_node_positions->getVector(0)->get1dView()[r_find] + step_length*N();
		const double y = boundary_node_positions->getVector(1)->get1dView()[r_find] + step_length*N();
		const double z = boundary_node_positions->getVector(2)->get1dView()[r_find] + step_length*N();
		mols.add_particle(x,y,z);
		tau = tau - log(U())/Ld;
	}
}

void PdeMoleculesCoupling::add_molecules_to_pde_test1(MoleculesSimple& mols,
		Pde& pde, const double overlap) {
	const std::vector<double>& x = mols.get_x();
	const std::vector<double>& y = mols.get_y();
	const std::vector<double>& z = mols.get_z();
	for (int i = 0; i < x.size(); ++i) {
		if (x[i] < 1.0-overlap) {
			pde.add_particle(x[i],y[i],z[i]);
			mols.remove_particle(i);
		}
	}
}




