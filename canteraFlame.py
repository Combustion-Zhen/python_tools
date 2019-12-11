import numpy as np
from scipy import special
import cantera as ct

def BilgerMixtureFraction( flame, fuel, oxidizer ):

    zFuel = BilgerSpecificMoleNumber( fuel )
    zOxidizer = BilgerSpecificMoleNumber( oxidizer )
    v = VectorMixtureFractionForMassFraction( fuel, oxidizer )

    Z = np.dot( flame.Y.transpose(), v )
    Z -= zOxidizer / ( zFuel - zOxidizer )

    return Z

def BilgerSpecificMoleNumber( gas ):

    zC = gas.elemental_mass_fraction('C')/gas.atomic_weight('C')
    zH = gas.elemental_mass_fraction('H')/gas.atomic_weight('H')
    zO = gas.elemental_mass_fraction('O')/gas.atomic_weight('O')

    z = 2.*zC + 0.5*zH - zO

    return z

def LagrangianFlameIndex( flame, fuel, oxidizer, 
                          speciesList = ['CO2', 'CO', 'H2O', 'H2'],
                          betaLean = 5., betaRich = 0.291 ):

    C, D, R = TransportBudget( flame )

    vectorMixtureFraction = VectorMixtureFractionForMassFraction( 
            fuel, oxidizer )
    vectorProgressVariable = VectorProgressVariableForMassFraction( 
            flame.gas, speciesList )

    fluxMixtureFraction = np.dot( D.transpose(), vectorMixtureFraction )
    fluxProgressVariable = np.dot( D.transpose(), vectorProgressVariable )

    magFluxMixtureFraction = np.absolute( fluxMixtureFraction )
    magFluxProgressVariable = np.absolute( fluxProgressVariable )

    flameMixtureFraction = BilgerMixtureFraction( flame, fuel, oxidizer )
    flameBeta =  ( 0.5 * ( betaLean - betaRich )
                  *special.erfc( ( flameMixtureFraction - 0.06 ) / 0.01 ) 
                 ) + betaRich
    magFlameBeta = np.absolute( flameBeta )

    indexLagrangian = ( magFluxProgressVariable 
                       - magFlameBeta * magFluxMixtureFraction 
                      ) / ( magFluxProgressVariable 
                           + magFlameBeta * magFluxMixtureFraction )
    return indexLagrangian

def Le( flame ):

    D = flame.mix_diff_coeffs
    alpha = flame.thermal_conductivity / (flame.density*flame.cp)
    Le = np.empty(D.shape)
    for i, D_spe in enumerate(D):
        Le[i] = alpha / D_spe

    return Le

def ProgressVariable( flame, speciesList ):

    v = VectorProgressVariableForMassFraction( flame.gas, speciesList )

    c = np.dot( flame.Y.transpose(), v )

    return c

def ProgressVariableReactionRate( flame, speciesList ):

    v = VectorProgressVariableForMassFraction( flame.gas, speciesList )

    massProductionRates = ( flame.net_production_rates.transpose() 
                           * flame.gas.molecular_weights )

    omega = np.dot( massProductionRates, v )

    return omega

def ScalarDissipationRateMixtureFraction( flame, fuel, oxidizer ):

    mixtureFraction = BilgerMixtureFraction( flame, fuel, oxidizer )

    mixtureFractionGradient = np.gradient( mixtureFraction, flame.grid )

    flame.transport_model = 'UnityLewis'

    chi = 2. * flame.mix_diff_coeffs[0] * np.square( mixtureFractionGradient )

    return chi

def StoichiometricMixtureFraction( fuel, oxidizer ):

    zFuel = BilgerSpecificMoleNumber( fuel )
    zOxidizer = BilgerSpecificMoleNumber( oxidizer )

    nu = - zFuel / zOxidizer

    Zst = 1. / ( 1. + nu )

    return Zst

def StoichiometricNu( gas, stream ):

    atom_list = ['C', 'H', 'O']
    atom_rate = [1, 0.25, -0.5]

    nu_stream = 0.

    for k, v in stream.items():

        index = gas.species_index(k)

        nu = 0.
        for i, atom in enumerate(atom_list):
            if atom in gas.element_names:
                nu += gas.n_atoms(index, atom) * atom_rate[i]

        nu_stream += nu * v

    return nu_stream

def StoichiometricNuOxy( gas, fuel ):

    atom_list = ['C', 'H', 'O']
    atom_rate = [1, 0.25, -0.5]

    index = gas.species_index(fuel)

    nu = 0.
    for i, atom in enumerate(atom_list):
        if atom in gas.element_names:
            nu += gas.n_atoms(index, atom) * atom_rate[i]

    return nu

def TakenoIndex( flame, fuelName, oxidizerName ):

    indexFuel = flame.gas.species_index( fuelName )
    indexOxidizer = flame.gas.species_index( oxidizerName )

    gradientFuel = np.gradient( flame.Y[indexFuel], flame.grid )
    gradientOxidizer = np.gradient( flame.Y[indexOxidizer], flame.grid )

    indexTakeno = gradientFuel * gradientOxidizer
    indexTakeno /= np.maximum( np.absolute( indexTakeno ), 
                               np.finfo( float ).resolution )

    return indexTakeno

def TransportBudget( flame ):

    gradientMassFraction = np.gradient( flame.Y, flame.grid, axis = 1 )

    convection = flame.density * flame.u * gradientMassFraction

    reaction = ( flame.net_production_rates.transpose() 
                * flame.gas.molecular_weights ).transpose()

    diffusion = convection - reaction

    return convection, diffusion, reaction

def TwoStreamsMixture( gas, fuel, oxidizer, phi ):

    # get stoichiometric coefficient

    # fuel stream
    nu_fuel = StoichiometricNu( gas, fuel )
    if nu_fuel <= 0.:
        raise ValueError('Fuel stream')

    # oxidizer stream
    nu_oxidizer = StoichiometricNu( gas, oxidizer )
    if nu_oxidizer >= 0.:
        raise ValueError('Oxidizer stream')

    nu = -nu_fuel/nu_oxidizer

    # construct mixture

    mixture = {}
    # add fuel
    for k, v in fuel.items():
        mixture[k] = v * phi
    # add oxidizer
    for k, v in oxidizer.items():
        if k in mixture.keys():
            mixture[k] += v * nu
        else:
            mixture[k] = v * nu

    return mixture

def VectorMixtureFractionForMassFraction( fuel, oxidizer ):

    zFuel = BilgerSpecificMoleNumber( fuel )
    zOxidizer = BilgerSpecificMoleNumber( oxidizer )

    mixtureFractionDenominator = zFuel - zOxidizer

    v = np.zeros( fuel.n_species )
    for i in range( fuel.n_species ):
        v[i] = 2.*fuel.n_atoms(i,'C') \
              +0.5*fuel.n_atoms(i,'H') \
              -fuel.n_atoms(i,'O')

    v /= fuel.molecular_weights
    v /= mixtureFractionDenominator

    return v

def VectorProgressVariableForMassFraction( gas, speciesList ):

    v = np.zeros( gas.n_species )

    for species in speciesList :
        v[gas.species_index(species)] = 1.

    return v

def FlameThermalThickness( x, temp ):

    temp_gradient = np.gradient(temp, x)
    temp_gradient_max = temp_gradient.max()
    delta = ( temp[-1] - temp[0] ) / temp_gradient_max

    return delta

def FlameConsumptionSpeed( flame, fuel ):

    # check the fuel info
    if isinstance( fuel, str ):
        # single component
        fuel_list = [fuel,]
    elif isinstance( fuel, list ):
        fuel_list = fuel

    fuel_rate = np.zeros( len(fuel_list) )
    fuel_mass = np.zeros( len(fuel_list) )

    for i, s in enumerate(fuel_list):

        # get species index
        index = flame.gas.species_index( s )

        # calculate fuel consumption
        fuel_rate[i] = - ( np.trapz(flame.net_production_rates[index],
                                    flame.grid)
                          *flame.gas.molecular_weights[index] )

        # fuel mass fraction difference
        fuel_mass[i] = flame.Y[index, 0] - flame.Y[index,-1]

    fuel_rate_sum = np.sum( fuel_rate )
    fuel_mass_sum = np.sum( fuel_mass )

    sc = fuel_rate_sum / ( flame.density[0] * fuel_mass_sum )

    return sc

def FuelConsumptionRate( flame, fuel ):

    # check the fuel info
    if isinstance( fuel, str ):
        # single component
        fuel_list = [fuel,]
    elif isinstance( fuel, list ):
        fuel_list = fuel

    fuel_rate = np.zeros((len(fuel_list), flame.T.size))

    for i, s, in enumerate(fuel_list):

        # get species index
        index = flame.gas.species_index( s )

        fuel_rate[i] = (-flame.net_production_rates[index]
                        *flame.gas.molecular_weights[index] )

    fuel_consumption_rate = np.sum( fuel_rate, axis=0 ) 

    return fuel_consumption_rate


