package za.redbridge.experiment.MultiObjective;

import org.encog.ml.ea.genome.Genome;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.NEATGenome;
import org.encog.neural.neat.training.NEATLinkGene;
import org.encog.neural.neat.training.NEATNeuronGene;
import za.redbridge.experiment.NEATM.FactorNEATMGenome;

import java.util.List;
import java.util.Random;

public class FactorMultiObjectiveNEATMGenome extends FactorNEATMGenome
{
    /**
     * {@inheritDoc}
     */
    @Override
    public NEATGenome factor() {
        return new MultiObjectiveNEATMGenome();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Genome factor(final Genome other) {
        return new MultiObjectiveNEATMGenome((MultiObjectiveNEATMGenome) other);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NEATGenome factor(final List<NEATNeuronGene> neurons,
                             final List<NEATLinkGene> links, final int inputCount,
                             final int outputCount) {
        return new MultiObjectiveNEATMGenome(neurons, links, inputCount, outputCount);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NEATGenome factor(final Random rnd, final NEATPopulation pop,
                             final int inputCount, final int outputCount,
                             final double connectionDensity) {
        return new MultiObjectiveNEATMGenome(rnd, pop, outputCount,
                connectionDensity);
    }
}
