/* 
 * python.cpp
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
 *  Created on: Feb 9, 2013
 *      Author: mrobins
 */


#include <boost/python.hpp>
#include "Pde.h"
#include "Species.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(pde_bd) {
	class_<Pde>("Pde", init<const char*, const double>())
			.def("integrate",&Pde::integrate)
			.def("add_particle",&Pde::add_particle)
			.def("init", &Pde::init,
					return_value_policy<manage_new_object>() )
			.staticmethod("init")
			;
    class_<Species>("Species");
}


