package za.redbridge.experiment.NEATM.training.opp;

import org.encog.mathutil.randomize.RangeRandomizer;
import org.encog.ml.ea.genome.Genome;
import org.encog.neural.neat.NEATNeuronType;
import org.encog.neural.neat.training.NEATGenome;
import org.encog.neural.neat.training.NEATLinkGene;
import org.encog.neural.neat.training.NEATNeuronGene;
import org.encog.neural.neat.training.opp.NEATMutateRemoveLink;

import za.redbridge.experiment.NEATM.training.NEATMGenome;
import za.redbridge.experiment.NEATM.training.NEATMNeuronGene;

import java.util.Random;

/**
 * Remove link mutation that can also remove input nodes.
 *
 * Created by jamie on 2014/10/20.
 */
public class NEATMMutateRemoveLink extends NEATMutateRemoveLink {

    public NEATMMutateRemoveLink() {
    }

    /*
    @Override
    public void performOperation(final Random rnd, final Genome[] parents,
                                 final int parentIndex, final Genome[] offspring,
                                 final int offspringIndex) {

        System.out.println("performOperation");

        final NEATGenome target = obtainGenome(parents, parentIndex, offspring,
                offspringIndex);

        if (target.getLinksChromosome().size() < NEATMutateRemoveLink.MIN_LINK) {
            // don't remove from small genomes
            return;
        }

        // determine the target and remove
        final int index = RangeRandomizer.randomInt(0, target
                .getLinksChromosome().size() - 1);
        final NEATLinkGene targetGene = target.getLinksChromosome().get(index);
        target.getLinksChromosome().remove(index);

        // if this orphaned any nodes, then kill them too!
        if (!isNeuronNeeded(target, targetGene.getFromNeuronID())) {
            System.out.println("ann size: " + target.getInputCount());
            removeNeuron(target, targetGene.getFromNeuronID());
            System.out.println("deleting neuron...");
            System.out.println("ann size: " + target.getInputCount());
        }

        if (!isNeuronNeeded(target, targetGene.getToNeuronID())) {
            System.out.println("ann size: " + target.getInputCount());
            removeNeuron(target, targetGene.getToNeuronID());
            System.out.println("deleting neuron...");
            System.out.println("ann size: " + target.getInputCount());
        }
    }
    */


    @Override
    public boolean isNeuronNeeded(NEATGenome target, long neuronID) {
        for (NEATNeuronGene neuron : target.getNeuronsChromosome()) {
            if (neuron.getId() == neuronID) {
                if (neuron.getNeuronType() == NEATNeuronType.Bias
                        || neuron.getNeuronType() == NEATNeuronType.Output) {
                    return true;
                }
                break;
            }
        }

        // Now check to see if the neuron is used in any links
        for (final NEATLinkGene link : target.getLinksChromosome()) {
            if (link.getFromNeuronID() == neuronID) {
                return true;
            }
            if (link.getToNeuronID() == neuronID) {
                return true;
            }
        }

        return false;
    }

    @Override
    public void removeNeuron(NEATGenome target, long neuronID) {
        for (NEATNeuronGene neuron : target.getNeuronsChromosome()) {
            if (neuron.getId() == neuronID) {
                if (neuron.getNeuronType() == NEATNeuronType.Input) {
                    ((NEATMGenome) target).removeInputNeuron((NEATMNeuronGene) neuron);
                } else {
                    target.getNeuronsChromosome().remove(neuron);
                }
                break;
            }
        }
    }
}
