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
#include <iostream>

int main(int argc, char **argv) {
	using boost::format;
	using std::string;

	Mpi::init(argc,argv);
	const double dt = 0.001;
	const double dt_out = dt;
	const double dx = 0.25;
	const double D = 1.0;
	const double overlap = 0.1;
	const double dx2 = dx*dx;
	Pde p(dt,dx);
	MoleculesSimple m;
	PdeMoleculesCoupling c;
	for (int i = 0; i < 1000; ++i) {
		p.add_particle(0.5,0.5,0.5);
	}
	for (int i = 0; i < 1.0/dt_out; ++i) {
		std::stringstream filename_grid,filename_boundary,filename_molecules;
		filename_grid <<format("test%05d.pvtu")%i;
		filename_boundary<<format("testBoundary%05d.pvtu")%i;
		filename_molecules<<format("testMolecules%05d.pvtu")%i;

		Io::write_grid(filename_grid.str(),p.get_grid());
		Io::write_grid(filename_boundary.str(),p.get_boundary());
		Io::write_points(filename_molecules.str(),m.get_x(),m.get_y(),m.get_z());

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
}

