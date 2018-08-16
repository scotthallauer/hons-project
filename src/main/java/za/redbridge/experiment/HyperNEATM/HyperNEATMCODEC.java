package za.redbridge.experiment.HyperNEATM;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSteepenedSigmoid;
import org.encog.ml.MLMethod;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.ea.codec.GeneticCODEC;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.GeneticError;
import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.neural.hyperneat.substrate.SubstrateLink;
import org.encog.neural.hyperneat.substrate.SubstrateNode;
import org.encog.neural.neat.NEATCODEC;
import org.encog.neural.neat.NEATLink;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.training.NEATNeuronGene;
import za.redbridge.experiment.NEAT.NEATPopulation;
import za.redbridge.experiment.NEATM.ActivationSteepenedShiftedSigmoid;
import za.redbridge.experiment.NEATM.NEATMNetwork;
import za.redbridge.experiment.NEATM.sensor.SensorModel;
import za.redbridge.experiment.NEATM.sensor.SensorMorphology;
import za.redbridge.experiment.NEATM.sensor.SensorType;


import java.io.Serializable;
import java.util.*;

/**
 * CODEC FOR HYPERNEATM
 * Creates sensors from CPPN
 *
 * Created by Danielle and Alexander on 2018/06/20.
 */
public class HyperNEATMCODEC implements GeneticCODEC, Serializable {

    private double minWeight = 0.2;
    private double maxWeight = 5.0;

    /**
     * {@inheritDoc}
     */
    @Override
    public MLMethod decode(final Genome genome) {
        final NEATPopulation pop = (NEATPopulation) genome.getPopulation();
        final Substrate substrate = pop.getSubstrate();
        return decode(pop, substrate, genome);
    }

    public MLMethod decode(final NEATPopulation pop, final Substrate substrate, final Genome genome) {
        //obtain the CPPN
        final NEATCODEC neatCodec = new NEATCODEC();
        final NEATNetwork cppn = (NEATNetwork) neatCodec.decode(genome);


        //setup for creating a new ANN from CPPN and substrate
        final List<NEATLink> linkList = new ArrayList<NEATLink>();



        final double c = this.maxWeight / (1.0 - this.minWeight);
        final MLData input = new BasicMLData(cppn.getInputCount());

        //create map for each input node and sensor
        HashMap<Integer, HyperNEATMSensorBuilder> InputNodeSensorMap = new HashMap<Integer, HyperNEATMSensorBuilder>();

        for(SubstrateNode Inputnode: substrate.getInputNodes() ){
            InputNodeSensorMap.put(Inputnode.getId(), new HyperNEATMSensorBuilder(Inputnode.getId(),Inputnode.getLocation()));
        }

        //hashmap to convert values in seqeuntial order for the ANN compute.
        final Map<Integer, Integer> lookup = new HashMap<>();
        //keeps track conversion inputnode number to actual number for ANN
        int countInputNodes =1;

        // First create all of the non-bias links and create a list sensor morphology

        for (final SubstrateLink link : substrate.getLinks()) {
            final SubstrateNode source = link.getSource();
            final SubstrateNode target = link.getTarget();

            int index = 0;
            for (final double d : source.getLocation()) {
                input.setData(index++, d);
            }
            for (final double d : target.getLocation()) {
                input.setData(index++, d);
            }
            final MLData output = cppn.compute(input);

            double weight = output.getData(0);

            if (Math.abs(weight) > this.minWeight) {
                weight = (Math.abs(weight) - this.minWeight) * c * Math.signum(weight);
                linkList.add(new NEATLink(source.getId(), target.getId(), weight));
                //if is an input node
                if (InputNodeSensorMap.containsKey(source.getId())||InputNodeSensorMap.containsKey(target.getId())){
                    //input nodes never connected
                    int IDUsing=source.getId();
                    if (InputNodeSensorMap.containsKey(target.getId())){
                        IDUsing =target.getId();
                    }
                    //update the sensors values
                    HyperNEATMSensorBuilder sensorBuilder = InputNodeSensorMap.get(IDUsing);
                    sensorBuilder.addWeights((float) (output.getData(0)));
                    sensorBuilder.addFOVs((float) (output.getData(1)));
                    sensorBuilder.addOrientations((float) (output.getData(2)));
                    sensorBuilder.addRanges((float) (output.getData(3)));
                    sensorBuilder.addSensorTypes((float) (output.getData(4)));
                    //ensures no memory issues (look into this)
                    InputNodeSensorMap.put(IDUsing, sensorBuilder);

                    //now adding to lookup so has ID NUMBER
                    if (lookup.get(IDUsing)==null){
                        lookup.put(IDUsing, countInputNodes++);
                    }

                }



            }
        }

        // now create biased links

        input.clear();
        final int d = substrate.getDimensions();
        final List<SubstrateNode> biasedNodes = substrate.getBiasedNodes();
        for (final SubstrateNode target : biasedNodes) {
            for (int i = 0; i < d; i++) {
                input.setData(d + i, target.getLocation()[i]);
            }

            final MLData output = cppn.compute(input);

            double biasWeight = output.getData(5);
            if (Math.abs(biasWeight) > this.minWeight) {
                biasWeight = (Math.abs(biasWeight) - this.minWeight) * c
                        * Math.signum(biasWeight);
                linkList.add(new NEATLink(0, target.getId(), biasWeight));
            }
        }

        // check for invalid neural network
        if (linkList.size() == 0) {
            return null;
        }

        Collections.sort(linkList);

        //create sensor Morphology list
        List<SensorModel> sensorModelsList = new ArrayList<SensorModel>();



        //get sensor morphology
       boolean first = true;
        for(SubstrateNode Inputnode: substrate.getInputNodes() ){
           // @todo look into bottom proximity
            if(first) {
                SensorModel bottomProximity = new SensorModel(SensorType.BOTTOM_PROXIMITY);
                sensorModelsList.add(bottomProximity);
                first =false;
                continue;
            }
            HyperNEATMSensorBuilder build = InputNodeSensorMap.get(Inputnode.getId());
            if (build.isValidSensor()){ //checks if the input node has connections
                sensorModelsList.add(build.createSensorModel());
            }

        }
        for(SubstrateNode Outputnode: substrate.getOutputNodes() ){
            lookup.put(Outputnode.getId(),countInputNodes++);
        }
        final List<NEATLink> linkListFinal = new ArrayList<NEATLink>();
        for(NEATLink link: linkList ){
            if(link.getFromNeuron()==0){
                linkListFinal.add( new NEATLink(0, lookup.get(link.getToNeuron()), link.getWeight()));
            }
            else{
                linkListFinal.add( new NEATLink(lookup.get(link.getFromNeuron()), lookup.get(link.getToNeuron()), link.getWeight()));
            }
        }



        //convert list to array for constructor
        SensorModel[] sensorModels = sensorModelsList.toArray(new SensorModel[sensorModelsList.size()]);
        SensorMorphology morphology = new SensorMorphology(sensorModels);


        final ActivationFunction[] afs = new ActivationFunction[sensorModelsList.size()+substrate.getOutputNodes().size()+1];
        //Activation Function for the right output
        final ActivationFunction af = new ActivationSteepenedShiftedSigmoid();
        // all activation functions are the same
        for (int i = 0; i < afs.length; i++) {
            afs[i] = af;
        }
        final NEATMNetwork network = new NEATMNetwork(sensorModels.length, substrate.getOutputCount(), linkListFinal, afs,morphology);
        network.setActivationCycles(substrate.getActivationCycles());

        return network;

    }
    @Override
    public Genome encode(final MLMethod phenotype) {
        throw new GeneticError(
                "Encoding of a HyperNEAT network is not supported.");
    }
}
