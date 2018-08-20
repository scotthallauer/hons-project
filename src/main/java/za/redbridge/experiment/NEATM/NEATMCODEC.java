package za.redbridge.experiment.NEATM;

import com.sun.org.apache.xpath.internal.operations.Bool;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.ml.MLMethod;
import org.encog.ml.ea.codec.GeneticCODEC;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.GeneticError;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.neat.NEATLink;
import org.encog.neural.neat.NEATNeuronType;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.NEATLinkGene;
import org.encog.neural.neat.training.NEATNeuronGene;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import za.redbridge.experiment.NEATM.sensor.SensorModel;
import za.redbridge.experiment.NEATM.sensor.SensorMorphology;
import za.redbridge.experiment.NEATM.training.NEATMGenome;
import za.redbridge.experiment.NEATM.training.NEATMNeuronGene;

/**
 * Created by jamie on 2014/09/08.
 */
public class NEATMCODEC implements GeneticCODEC, Serializable {

    private static final long serialVersionUID = -5767773750949848124L;

    @Override
    public MLMethod decode(Genome genome) {
        final NEATMGenome neatGenome = (NEATMGenome) genome;
        final NEATPopulation pop = (NEATPopulation) neatGenome.getPopulation();
        final List<NEATNeuronGene> neuronsChromosome = neatGenome.getNeuronsChromosome();
        final List<NEATLinkGene> linksChromosome = neatGenome.getLinksChromosome();

        if (neuronsChromosome.get(0).getNeuronType() != NEATNeuronType.Bias) {
            throw new NeuralNetworkError(
                    "The first neuron must be the bias neuron, this genome is invalid.");
        }

        final List<NEATLink> links = new ArrayList<NEATLink>();
        final ActivationFunction[] afs = new ActivationFunction[neuronsChromosome.size()];

        for (int i = 0; i < afs.length; i++) {
            afs[i] = neuronsChromosome.get(i).getActivationFunction();
        }

        final Map<Long, Integer> lookup = new HashMap<>();
        //check if sensorExit
        final Map<Long, Boolean> sensorExist = new HashMap<>();
        for (int i = 0; i < linksChromosome.size(); i++) {
            final NEATLinkGene linkGene = linksChromosome.get(i);
            if (linkGene.isEnabled()) {
                sensorExist.put(linkGene.getFromNeuronID(),true);
            }

        }

        int keepTrackLookup=0;
        int inputCount =0;
        for (int i = 0; i < neuronsChromosome.size(); i++) {
            final NEATNeuronGene neuronGene = neuronsChromosome.get(i);
            //only make a new gene if is valid sensor
            if(sensorExist.get(neuronGene.getId())!=null||neuronGene.getNeuronType()!=NEATNeuronType.Input) {
                lookup.put(neuronGene.getId(), keepTrackLookup);
                keepTrackLookup++;
                if(neuronGene.getNeuronType()==NEATNeuronType.Input){
                    inputCount++;
                }
            }
        }

        // loop over connections

        for (int i = 0; i < linksChromosome.size(); i++) {
            final NEATLinkGene linkGene = linksChromosome.get(i);
            if (linkGene.isEnabled()) {
                if(lookup.get(linkGene.getFromNeuronID())==null|| lookup.get(linkGene.getToNeuronID())==null){
                    throw new NeuralNetworkError(
                            "ERROR DECODING PHENOME ");
                }
                links.add(new NEATLink(lookup.get(linkGene.getFromNeuronID()),
                        lookup.get(linkGene.getToNeuronID()), linkGene
                        .getWeight()));
            }

        }

        Collections.sort(links);

        // Create the sensor morphology

        final List<NEATMNeuronGene> inputNeurons = neatGenome.getInputNeuronsChromosome();
        SensorModel[] sensorModels = new SensorModel[inputCount];
        int countSensor =0;
        for (int i = 0; i < inputNeurons.size(); i++) {
            NEATMNeuronGene inputNeuron = inputNeurons.get(i);
            if(sensorExist.get(inputNeuron.getId())!=null) {

                sensorModels[countSensor] = inputNeuron.getSensorConfiguration().toSensorModel();
                countSensor++;
            }
        }

        SensorMorphology morphology = new SensorMorphology(sensorModels);

        NEATMNetwork network = new NEATMNetwork(inputCount, neatGenome.getOutputCount(), links,
                afs, morphology);

        network.setActivationCycles(pop.getActivationCycles());
        return network;
    }

    @Override
    public Genome encode(MLMethod phenotype) {
        throw new GeneticError("Encoding of a NEAT network is not supported.");
    }
}
