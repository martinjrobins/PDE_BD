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
#include <Teuchos_ArrayView.hpp>

Molecules::Molecules() {
	using Tpetra::Map;
	using Teuchos::ArrayView;
	my_rank = Mpi::mpiSession->getRank();
	num_procs = Mpi::mpiSession->getNProc();
	// Get the default communicator and Kokkos Node instance
	comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
	node = Tpetra::DefaultPlatform::getDefaultPlatform ().getNode ();
	gids.push_back(my_rank);
	map_allocate = rcp(new Map(num_procs,ArrayView(gids),0,comm,node));
	pos_allocate = rcp(new multivector_type(map_allocate,3));
	alive = rcp(new vector_int_type(map_allocate));
	num_local_particles = 0;
	num_global_particles = 0;
	num_allocated_particles = 1;
	map = rcp(new Map(num_procs));
	pos = pos_allocate->offsetViewNonConst(map,0);
}

void Molecules::add_particle(const ST x, const ST Y, const ST z) {
	if (num_local_particles == num_allocated_particles) {
		for (int i = num_allocated_particles; i < 2*num_allocated_particles; ++i) {
			gids.push_back(i*num_procs + my_rank);
		}
		/*
		 * allocate new space
		 */
		RCP<map_type> new_map_allocate = rcp(new Map(num_procs,ArrayView(gids),0,comm,node));
		RCP<multivector_type> new_pos_allocate = rcp(new multivector_type(map_allocate,3));
		RCP<vector_int_type> new_alive = rcp(new vector_int_type(map_allocate));
		/*
		 * copy old data across
		 */

	}
	pos
}

void Molecules::remove_particle(const GO i) {
}

void Molecules::diffuse(const double dt, const double D) {
}


