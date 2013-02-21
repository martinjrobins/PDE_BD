/*
 * MoleculesSimple.h
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
 *  Created on: 21 Feb 2013
 *      Author: robinsonm
 */

#ifndef MOLECULESSIMPLE_H_
#define MOLECULESSIMPLE_H_

#include <vector>
#include <trng/yarn2.hpp>
#include <trng/config.hpp>

class MoleculesSimple {
public:
	MoleculesSimple();
	void add_particle(const double x, const double Y, const double z);
	void remove_particle(const int i);
	void diffuse(const double dt, const double D);
	const std::vector<double>& get_x() {return x;}
	const std::vector<double>& get_y() {return y;}
	const std::vector<double>& get_z() {return z;}
private:
	std::vector<double> x,y,z;
	trng::yarn2 R;
};

#endif /* MOLECULESSIMPLE_H_ */