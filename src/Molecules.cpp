/* 
 * Molecules.cpp
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
 *  Created on: Feb 17, 2013
 *      Author: mrobins
 */

#include "Molecules.h"

Molecules::Molecules() {
	using Tpetra::Map;
	using Tpetra::ArrayView;
	const int my_rank = Mpi::mpiSession->getRank();
	const int num_procs = Mpi::mpiSession->getNProc();
	// Get the default communicator and Kokkos Node instance
	comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
	node = Tpetra::DefaultPlatform::getDefaultPlatform ().getNode ();
	gids_data.push_back(my_rank);
	map_data = rcp(new Map(num_procs,ArrayView(gids_data),0,comm,node));
	pos_data = rcp(new multivector_type(map_data,3));
}

void Molecules::add_particle(const ST x, const ST Y, const ST z) {
}

void Molecules::remove_particle(const GO i) {
}

void Molecules::diffuse(const double dt, const double D) {
}


