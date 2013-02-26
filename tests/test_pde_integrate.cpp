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
	const double dx = 0.25;
	Pde p(dt,dx);
	p.add_particle(0.5,0.5,0.5);
	for (int i = 0; i < 0.1/dt; ++i) {
		std::stringstream filename_grid,filename_boundary;
		filename_grid <<format("test%05d.pvtu")%i;
		filename_boundary<<format("testBoundary%05d.pvtu")%i;

		Io::write_grid(filename_grid.str(),p.get_grid());
		Io::write_grid(filename_boundary.str(),p.get_boundary());
		p.integrate(dt);
	}
}


