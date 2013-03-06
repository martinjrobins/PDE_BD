/*
 * PdeMoleculesCoupling.h
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

#ifndef PDEMOLECULESCOUPLING_H_
#define PDEMOLECULESCOUPLING_H_

#include "Pde.h"
#include "MoleculesSimple.h"
#include <boost/random.hpp>


class PdeMoleculesCoupling {
	typedef boost::mt19937  base_generator_type;
public:
	PdeMoleculesCoupling() {
		R.seed(time(NULL));
	}
	int generate_new_molecules(MoleculesSimple& mols, Pde& pde, const double dt, const double dA, const double D);
	void add_molecules_to_pde_test1(MoleculesSimple& mols,Pde& pde, const double overlap);
	void add_molecules_to_pde_test2(MoleculesSimple& mols,Pde& pde, const double overlap);

private:
	base_generator_type R;
};


#endif /* PDEMOLECULESCOUPLING_H_ */
