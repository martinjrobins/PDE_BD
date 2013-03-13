/* 
 * test_pde_constructor.cpp
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
 *  Created on: Feb 16, 2013
 *      Author: mrobins
 */

#include "Pde_bd.h"
#include "boost/format.hpp"
#include "zip.h"

#include <iostream>

int main(int argc, char **argv) {
	using boost::format;
	using std::string;

	Mpi::init(argc,argv);
	const double dt = 0.001;
	const double max_t = 20.0;
	const double dt_out = dt;
	const double dx = 0.1;
	const double D = 1.0;
	const double overlap = 0.1;
	const double dx2 = dx*dx;
	Pde p(dt,dx,1,4);
	MoleculesSimple m;
	PdeMoleculesCoupling c;
	for (int i = 0; i < 1000; ++i) {
		p.add_particle(0,0,0);
	}
	std::vector<double> pde_sizes,mol_sizes,t,patch_sizes;
	for (int i = 0; i < max_t/dt_out; ++i) {
		std::stringstream filename_grid,filename_boundary,filename_molecules;
		filename_grid <<format("test%05d.pvtu")%i;
		filename_boundary<<format("testBoundary%05d.pvtu")%i;
		filename_molecules<<format("testMolecules%05d.pvtu")%i;

		//Io::write_grid(filename_grid.str(),p.get_grid());
		//Io::write_grid(filename_boundary.str(),p.get_boundary());
		//Io::write_points(filename_molecules.str(),m.get_x(),m.get_y(),m.get_z());

		int Npde = p.get_total_number_of_particles();

		int Nmol_pde = 0;
		int Nmol = 0;

		std::for_each(m.get_x().begin(),m.get_x().end(), [&](double x) {
			if (x < 1.0) {
				Nmol_pde++;
			} else  {
				Nmol++;
			}
		});

		t.push_back(i*dt_out);
		pde_sizes.push_back(Npde+Nmol_pde);
		mol_sizes.push_back(Nmol);

		std::cout << *(pde_sizes.end()-1) << " molecules in pde subdomain and "<<
				     *(mol_sizes.end()-1) << " molecules in particle subdomain."<<std::endl;

		double Nmol_patch = 0;
		auto z = zip(m.get_x(),m.get_y(),m.get_z());
		typedef decltype(z) zip_type;
		std::for_each(z.begin(),z.end(), [&](zip_type::reference i) {
			const double x = i.get<0>();
			const double y = i.get<1>();
			const double z = i.get<2>();
			const double s = 0.1;
			if ((x > 1.0) && (x < 1.0+s) && (y > 0.2) && (y < 0.2+s) && (z > 0.2) && (z < 0.2+s)) {
				Nmol_patch++;
			}
		});
		patch_sizes.push_back(Nmol_patch);

		const int iterations = int(dt_out/dt + 0.5);
		const double actual_dt = iterations*dt;
		for (int j = 0; j < iterations; ++j) {
			p.integrate(dt);
			c.generate_new_molecules(m, p, dt, dx2, D);
			m.diffuse(dt,D);
			m.reflective_boundaries(0,2,0,1,0,1);
			c.add_molecules_to_pde_test1(m, p, overlap);
		}
	}
	std::vector<std::vector<double>* > columns;
	columns.push_back(&t);
	columns.push_back(&pde_sizes);
	columns.push_back(&mol_sizes);
	columns.push_back(&patch_sizes);
	Io::write_column_vectors("number_of_molecules","#time num_pde num_particle num_patch",columns);
}


