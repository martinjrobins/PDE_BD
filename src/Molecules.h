/* 
 * Molecules.h
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

#ifndef MOLECULES_H_
#define MOLECULES_H_

#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_DefaultPlatform.hpp"

using Teuchos::RCP;
using Teuchos::rcp;

class Molecules {
	typedef double ST;
	typedef int    LO;
	typedef int    GO;
	typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType  Node;

	typedef Tpetra::MultiVector<ST, LO, GO, Node>  multivector_type;
	typedef Tpetra::Vector<ST, LO, GO, Node>       vector_type;

	typedef Tpetra::Map<LO, GO, Node>         map_type;
	typedef Tpetra::Export<LO, GO, Node>      export_type;
	typedef Tpetra::Import<LO, GO, Node>      import_type;
public:
	Molecules();
	void add_particle(const ST x, const ST Y, const ST z);
	void remove_particle(const GO i);
	void diffuse(const double dt, const double D);
private:

	RCP<const Teuchos::Comm<int> > comm;
	RCP<Node> node;

	RCP<multivector_type> pos_data, pos_view;
	RCP<Tpetra::Map> map_data,map_view;
	std::vector<GO> gids_data;
	LO num_local_particles;
	GO num_global_particles;

};

#endif /* MOLECULES_H_ */
