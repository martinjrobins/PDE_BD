/*
 * MoleculesSimple.cpp
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

#include "MoleculesSimple.h"
#include <boost/bind.hpp>
#include <trng/normal_dist.hpp>
#include <algorithm>
#include <time.h>

MoleculesSimple::MoleculesSimple() {
	R.seed(time(NULL));
}

void MoleculesSimple::add_particle(const double x_new, const double y_new,
		const double z_new) {
	x.push_back(x_new); y.push_back(y_new); z.push_back(z_new);
}

void MoleculesSimple::remove_particle(const int i) {
	x[i] = *(x.end()-1);
	y[i] = *(y.end()-1);
	z[i] = *(z.end()-1);
	x.pop_back();y.pop_back();z.pop_back();
}


void MoleculesSimple::diffuse(const double dt, const double D) {
	trng::normal_dist<double> N(0,1);
	const double rms_step = sqrt(2.0*D*dt);

	auto diffuse = [&](double d){ return d + rms_step*N(R); };

	std::transform(x.begin(), x.end(), x.begin(), diffuse);
	std::transform(y.begin(), y.end(), y.begin(), diffuse);
	std::transform(z.begin(), z.end(), z.begin(), diffuse);
}


