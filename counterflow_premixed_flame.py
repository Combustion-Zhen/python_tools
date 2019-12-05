#%%
"""

An counterflow premixed flame

to get flame speed/scalar profile resond to strain rate/pressure

strain rate is calculated based on Niemann et al., CNF, 2015

momentum balance $\rho_1U_1^2=\rho_2U_2^2$ is imposed to put the flame in center

Zhen Lu

"""

import numpy as np
import cantera as ct
import argparse
from filename import params2name

def counterflow_premix_flame(
    mech='chem.cti', transport='Mix', fuel_name='H2', phi=0.6, T_in=300., p=1.,
    a=1000., width=0.01, solution=None):

    # construct case name
    flame_params = {}
    flame_params['F'] = fuel_name
    flame_params['phi'] = phi
    flame_params['T'] = T_in
    flame_params['p'] = p
    flame_params['a'] = a

    flame_name = params2name(flame_params)

    # Create gas object
    gas = ct.Solution(mech)

    p *= ct.one_atm

    # oxidizer
    oxy = {'O2':1., 'N2':3.76}
    
    # get composition
    fuel_index = gas.species_index(fuel_name)

    # stoichiometric coefficient for oxidizer
    stoich_nu = 0.
    if 'C' in gas.element_names:
        stoich_nu += gas.n_atoms(fuel_index, 'C')
    if 'H' in gas.element_names:
        stoich_nu += gas.n_atoms(fuel_index, 'H')/4.
    if 'O' in gas.element_names:
        stoich_nu -= gas.n_atoms(fuel_index, 'O')/2.

    comp = {}
    comp[fuel_name] = 1.
    for k, v in oxy.items():
        comp[k] = v*stoich_nu/phi

    gas.TPX = T_in, p, comp
    rho_u = gas.density

    gas.equilibrate('HP')
    rho_b = gas.density

    gas.TPX = T_in, p, comp

    # get inlet velocity based on the strain rate
    # $a_1=\dfrac{2U_1}{L}\left(1+\dfrac{U_2\sqrt{\rho_2}}{U_1\sqrt{\rho_1}}\right)$
    # $a_2=\dfrac{2U_2}{L}\left(1+\dfrac{U_1\sqrt{\rho_1}}{U_2\sqrt{\rho_2}}\right)$
    # with $\rho_1 U_1^2 = \rho_2 U_2^2$
    # $a_1=\dfrac{4U_1}{L}$ $a_2=\dfrac{4U_2}{L}$
    # set stream 1 and 2 for unburnt and equilibrium status respectively
    v_u = a * width / 4.
    v_b = np.sqrt( rho_u*np.square(v_u) / rho_b )

    # mass rate
    m_u = rho_u * v_u
    m_b = rho_b * v_b

    # Create flame object
    f = ct.CounterflowPremixedFlame(gas=gas, width=width)

    f.transport_model = transport
    f.P = p
    f.reactants.mdot = m_u
    f.products.mdot = m_b

    #f.set_refine_criteria(ratio=2.0, slope=0.015, curve=0.01, prune=0.002)
    #f.set_refine_criteria(ratio=2.0, slope=0.1, curve=0.2, prune=0.02)
    f.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02, prune=0.002)
    f.set_max_grid_points(f.flame, 5000)

    # load saved case if presented
    if solution is not None:

        f.restore(solution, loglevel=0)

        # scaling of saved solution
        solution_width = f.grid[-1] - f.grid[0]
        width_factor = width / solution_width

        solution_a = 4.*f.u[0]/solution_width
        a_factor = a / solution_a

        normalized_grid = f.grid / solution_width

        u_factor = a_factor * width_factor

        # update solution initialization following Fiala & Sattelmayer
        f.flame.grid = normalized_grid * width
        f.set_profile('u', normalized_grid, f.u*u_factor)
        f.set_profile('V', normalized_grid, f.V*a_factor)
        f.set_profile('lambda', normalized_grid, f.L*np.square(a_factor))

        f.reactants.mdot = m_u
        f.products.mdot = m_b
    else:

        f.set_initial_guess()

    f.solve(loglevel=0, auto=True)

    HRR = f.heat_release_rate

    idx = HRR.argmax()

    if HRR[idx] > 1000 :

        f.save('{}.xml'.format(flame_name))

        if f.u[idx] > 0 :

            return 0

        else :

            return 2

    else:

        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--mechanism',
        default = 'chem.cti',
        type = str,
        help = 'reaction mechanism')

    parser.add_argument(
        '-t', '--transport',
        default = 'Mix',
        type = str,
        help = 'transport model')

    parser.add_argument(
        '-f', '--fuel',
        default = 'H2',
        type = str,
        help = 'fuel name')

    parser.add_argument(
        '-a', '--strain',
        default = 1000.,
        type = float,
        help = 'mean strain rate (1/s)')

    parser.add_argument(
        '-w', '--width',
        default = 0.01,
        type = float,
        help = 'flame domain (m)')

    parser.add_argument(
        '-p', '--pressure',
        default = 1.,
        type = float,
        help = 'pressure (atm)')

    parser.add_argument(
        '--phi',
        default = 0.6,
        type = float,
        help = 'equivalence ratio')

    parser.add_argument(
        '--Tin',
        default = 300.,
        type = float,
        help = 'temperature of the unburnt stream')

    parser.add_argument(
        '-s', '--solution',
        default = None,
        type = str,
        help = 'restart solution')

    args = parser.parse_args()

    counterflow_premix_flame(
        mech = args.mechanism,
        transport = args.transport,
        fuel_name = args.fuel,
        phi = args.phi,
        T_in = args.Tin,
        p = args.pressure,
        a = args.strain,
        width = args.width,
        solution = args.solution)
