package za.redbridge.experiment.NEATM;

import org.encog.ml.ea.species.BasicSpecies;
import org.encog.neural.hyperneat.FactorHyperNEATGenome;
import org.encog.neural.hyperneat.HyperNEATCODEC;
import org.encog.neural.hyperneat.HyperNEATGenome;
import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.neural.neat.FactorNEATGenome;
import org.encog.neural.neat.NEATCODEC;
import org.encog.neural.neat.training.NEATGenome;
import org.encog.neural.neat.training.NEATInnovationList;

import java.util.Random;

import za.redbridge.experiment.HyperNEATM.HyperNEATMCODEC;
import za.redbridge.experiment.NEATM.sensor.SensorType;
import za.redbridge.experiment.NEAT.NEATPopulation;

/**
 * Created by jamie on 2014/09/08.
 */
public class NEATMPopulation extends NEATPopulation
{

    private static final long serialVersionUID = -6647644833955733411L;

    /**
     * An empty constructor for serialization.
     */
    public NEATMPopulation()
    {

    }

    /**
     * Construct a starting NEATM population. This does not generate the initial
     * random population of genomes.
     *
     * @param outputCount    The output neuron count.
     * @param populationSize The population size.
     */
    public NEATMPopulation(int outputCount, int populationSize)
    {
        super(SensorType.values().length, outputCount, populationSize); // inputCount = SensorType.values().length
    }                                                                   // This is because initial population begins with one of each sensor type.


    /**
     * Construct a starting HyperNEATM population. This does not generate the
     * initial random population of genomes.
     *
     * @param theSubstrate
     *            The substrate ID.
     * @param populationSize
     *            Number of CPPNs in each generation.
     */
    public NEATMPopulation(final Substrate theSubstrate, final int populationSize)
    {
        super(theSubstrate, populationSize);
        setInputCount(getSubstrate().getInputCount());
        setOutputCount(getSubstrate().getOutputCount());
    }


    @Override
    public void reset()
    {

        // create the genome factory
        if (isHyperNEAT())                                  // Just checks if NEAT(M)Population has a valid substrate.
        {
            setCODEC(new HyperNEATMCODEC());
            setGenomeFactory(new FactorHyperNEATGenome());
        }
        else
        {
            setCODEC(new NEATMCODEC());
            setGenomeFactory(new FactorNEATMGenome());
        }


        // create the new genomes
        getSpecies().clear();

        // reset counters
        getGeneIDGenerate().setCurrentID(1);
        getInnovationIDGenerate().setCurrentID(1);

        final Random rnd = getRandomNumberFactory().factor();

        // create one default species
        final BasicSpecies defaultSpecies = new BasicSpecies();
        defaultSpecies.setPopulation(this);

        // create the initial population
        for (int i = 0; i < getPopulationSize(); i++)
        {
            final NEATGenome genome = getGenomeFactory().factor(rnd, this,
                    getInputCount(), getOutputCount(), getInitialConnectionDensity());
            defaultSpecies.add(genome);
        }
        defaultSpecies.setLeader(defaultSpecies.getMembers().get(0));
        getSpecies().add(defaultSpecies);

        // create initial innovations
        setInnovations(new NEATInnovationList(this));
    }

}
