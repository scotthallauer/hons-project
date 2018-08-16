package za.redbridge.experiment.MultiObjective;

import org.encog.ml.ea.genome.Genome;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.NEATGenome;
import org.encog.neural.neat.training.NEATLinkGene;
import org.encog.neural.neat.training.NEATNeuronGene;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

/**
 * Multi-Objective HyperNEAT Genome
 * Written by Danielle and ALex
 *
 */

public class FactorMultiObjectiveHyperNEATGenome extends org.encog.neural.hyperneat.FactorHyperNEATGenome implements Serializable
{
    /**
     * {@inheritDoc}
     */
    @Override
    public NEATGenome factor() {
        return new MultiObjectiveHyperNEATGenome();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Genome factor(final Genome other) {
        return new MultiObjectiveHyperNEATGenome((MultiObjectiveHyperNEATGenome) other);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NEATGenome factor(final List<NEATNeuronGene> neurons,
                             final List<NEATLinkGene> links, final int inputCount,
                             final int outputCount) {
        return new MultiObjectiveHyperNEATGenome(neurons, links, inputCount, outputCount);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NEATGenome factor(final Random rnd, final NEATPopulation pop,
                             final int inputCount, final int outputCount,
                             final double connectionDensity) {
        return new MultiObjectiveHyperNEATGenome(rnd, pop, inputCount, outputCount,
                connectionDensity);
    }
}
