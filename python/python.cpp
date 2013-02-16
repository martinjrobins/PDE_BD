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
#include "Pde_bd.h"


using namespace boost::python;

void pde_init(boost::python::list& py_argv) {
	using boost::python::len;

	int argc = len(py_argv);
	char **argv = new char*[len(py_argv)];
	for (int i = 0; i < len(py_argv); ++i) {
		std::string this_argv = boost::python::extract<std::string>(py_argv[i]);
		argv[i] = new char[this_argv.size()];
		std::strcpy(argv[i],this_argv.c_str());
	}
	Pde::init(argc,argv);
	for (int i = 0; i < len(py_argv); ++i) {
		delete argv[i];
	}
	delete argv;
}



BOOST_PYTHON_MODULE(pde_bd) {
	def("init", pde_init);
	class_<Pde>("Pde", init<const double, const double>())
			.def("integrate",&Pde::integrate)
			.def("add_particle",&Pde::add_particle)
			;
    class_<Species>("Species");
}


