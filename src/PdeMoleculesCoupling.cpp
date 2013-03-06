/*
 * PdeMoleculesCoupling.cpp
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

#include "PdeMoleculesCoupling.h"
#include <MatrixMarket_Tpetra.hpp>
//#include <math.h>

#include <Tpetra_RTI.hpp>
#include <Intrepid_FieldContainer.hpp>
#include <numeric>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>

int find_first(const double to_find, const std::vector<double>& to_find_in) {
	if (to_find_in[0] >= to_find) return 0;
	for (int i = 1; i < to_find_in.size(); ++i) {
		if (to_find_in[i] >= to_find) {
			if (to_find_in[i]-to_find < to_find-to_find_in[i-1]) {
				return i;
			} else {
				return i-1;
			}
		}
	}
}


int PdeMoleculesCoupling::generate_new_molecules(MoleculesSimple& mols,
		Pde& pde, const double dt, const double dA, const double D) {
	RCP<Pde::vector_type> boundary_node_values = pde.get_boundary_node_values();
	RCP<Pde::multivector_type> boundary_node_positions = pde.get_boundary_node_positions();

	std::vector<double> bnv(boundary_node_values->getLocalLength());
	std::vector<double> bnv_cumsum(boundary_node_values->getLocalLength());
	boundary_node_values->get1dCopy(Teuchos::ArrayView<double>(bnv));
	for (int i = 0; i < boundary_node_values->getLocalLength(); ++i) {
		const double x = boundary_node_positions->getVector(0)->get1dView()[i];
		const double y = boundary_node_positions->getVector(1)->get1dView()[i];
		const double z = boundary_node_positions->getVector(2)->get1dView()[i];
		if ((y==0)||(y==1)) {
			bnv[i] /= 2.0;
		}
		if ((z==0)||(z==1)) {
			bnv[i] /= 2.0;
		}
	}
	//std::replace_if(bnv.begin(),bnv.end(),[](double d){return d<0;},0);
	std::partial_sum(bnv.begin(),bnv.end(),bnv_cumsum.begin());

	const double sum_values = *(bnv_cumsum.end()-1);
	const double Ld = dA*sum_values/dt;

	if (Ld <= 0) return 0;

	boost::variate_generator<base_generator_type&, boost::uniform_real<> >
				U(R,boost::uniform_real<>(0,1));
	boost::variate_generator<base_generator_type&, boost::normal_distribution<> >
				N(R,boost::normal_distribution<>(0,1));

//	std::cout << "x vector = " << std::endl;
//	boundary_node_positions->getVector(0)->print(std::cout);
//	Tpetra::MatrixMarket::Writer<Pde::vector_type>::writeDense (std::cout, boundary_node_positions->getVector(0));

	double tau = -log(U())/Ld;
	const int Npde = pde.get_total_number_of_particles();
	int Ngenerated = 0;
	while (tau < dt) {
		const double r = U()*sum_values;
		std::vector<double>::iterator it = std::find_first_of(bnv_cumsum.begin(),bnv_cumsum.end(),&r,&r+1,
					[](double a, double b){return a >= b;});
//		if ((it != bnv_cumsum.begin()) && (*it-r > r-*(it-1))) {
//			it--;
//		}
		const int r_find = it-bnv_cumsum.begin();
		//const int r_find = find_first(r,bnv_cumsum);
		const double step_length = sqrt(2.0*D*(dt-tau));
		const double x = boundary_node_positions->getVector(0)->get1dView()[r_find] + step_length*N();
		const double y = boundary_node_positions->getVector(1)->get1dView()[r_find] + step_length*N();
		const double z = boundary_node_positions->getVector(2)->get1dView()[r_find] + step_length*N();
		mols.add_particle(x,y,z);
		tau = tau - log(U())/Ld;
		Ngenerated++;
	}

	pde.rescale((Npde-Ngenerated+Ld*dt)/Npde);
	return Ngenerated;
}

struct equal_one {
	bool operator()(boost::tuple<double,double,double,int> p) {
		return boost::get<3>(p)==1;
	}
};

void PdeMoleculesCoupling::add_molecules_to_pde_test1(MoleculesSimple& mols,
		Pde& pde, const double overlap) {

	const std::vector<double>& x = mols.get_x();
	const std::vector<double>& y = mols.get_y();
	const std::vector<double>& z = mols.get_z();
	std::vector<int> points_added(x.size(),0);
	for (int i = 0; i < x.size(); ++i) {
		if (x[i] < 1-overlap) {
			pde.add_particle(x[i],y[i],z[i]);
			points_added[i] = 1;
		}
	}

//	std::vector<int> points_added;
//	pde.add_particles(points_added,x,y,z);
	mols.remove_particles(points_added);

//	typedef boost::tuple<double,double,double,int> tuple_type;
//	typedef boost::tuple<std::vector<double>::iterator,std::vector<double>::iterator,
//			       std::vector<double>::iterator,std::vector<int>::iterator> tuple_iterator_type;
//	typedef boost::zip_iterator<tuple_iterator_type> zip_type;
//
//	zip_type b = boost::make_zip_iterator(boost::make_tuple(x.begin(),y.begin(),z.begin(),points_added.begin()));
//	zip_type e = boost::make_zip_iterator(boost::make_tuple(x.end(),y.end(),z.end(),points_added.end()));
//
////	zip_type to_delete = std::partition(b,e,
////				   [](tuple_type p) {return boost::get<3>(p)==1;} );
//	zip_type to_delete = std::partition(b,e,equal_one());
//
//	x.erase(boost::get<0>(to_delete.get_iterator_tuple()),x.end());
//	y.erase(boost::get<1>(to_delete.get_iterator_tuple()),y.end());
//	z.erase(boost::get<2>(to_delete.get_iterator_tuple()),y.end());

}




