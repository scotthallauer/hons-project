package za.redbridge.experiment;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SynchronizedDescriptiveStatistics;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.neural.neat.NEATNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sim.display.Console;
import za.redbridge.experiment.NEATM.NEATMNetwork;
import za.redbridge.experiment.NEATM.sensor.SensorMorphology;
import za.redbridge.experiment.NEATM.sensor.SensorType;
import za.redbridge.experiment.NEATM.sensor.parameter.spec.ParameterType;
import za.redbridge.experiment.NEATM.sensor.parameter.spec.Range;
import za.redbridge.simulator.Simulation;
import za.redbridge.simulator.SimulationGUI;
import za.redbridge.simulator.config.SimConfig;
import za.redbridge.simulator.factories.HomogeneousRobotFactory;
import za.redbridge.simulator.factories.RobotFactory;
import za.redbridge.simulator.phenotype.Phenotype;
import za.redbridge.simulator.sensor.AgentSensor;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.ArrayList;

/**
 * Test runner for the simulation.
 * <p>
 * Created by jamie on 2014/09/09.
 */
public class ScoreCalculator implements CalculateScore
{

    private static final Logger log = LoggerFactory.getLogger(ScoreCalculator.class);

    private final SimConfig simConfig;
    private final int trialsPerIndividual;
    private final SensorMorphology sensorMorphology;

    private final DescriptiveStatistics TimeTakenStats = new SynchronizedDescriptiveStatistics();
    private final DescriptiveStatistics scoreStats = new SynchronizedDescriptiveStatistics();
    private final DescriptiveStatistics sensorStats;
    private final DescriptiveStatistics neuralStats = new SynchronizedDescriptiveStatistics();

    private boolean hyperNEATM;

    private final HashMap<String, DescriptiveStatistics[]> sensorParam= new HashMap<>();
    public final String[] names = {"ColourProximitySensor", "LowResCameraSensor", "ProximitySensor", "UltrasonicSensor"};



    public ScoreCalculator(SimConfig simConfig, int trialsPerIndividual,
                           SensorMorphology sensorMorphology, boolean hyperNEATM)
    {
        this.simConfig = simConfig;
        this.trialsPerIndividual = trialsPerIndividual;
        this.sensorMorphology = sensorMorphology;
        this.hyperNEATM = hyperNEATM;

        // If fixed morphology then don't record sensor stats
        this.sensorStats = isEvolvingMorphology() ? new SynchronizedDescriptiveStatistics() : null;

        for(int j =0;j<names.length;j++) {
            DescriptiveStatistics[] stats = new DescriptiveStatistics[3];
            for (int i = 0; i < 3; i++) // 3 parameters per sensor
            {
                stats[i] = new DescriptiveStatistics();
            }
            sensorParam.put(names[j],stats);

        }
    }

    /*
    * Caclulates the total number of nodes and links for a given Network
    */

    public int getNeuralEnergyCost(MLMethod method) {

        NEATNetwork network = (NEATNetwork) method;
        ArrayList<Integer> networkNodes = new ArrayList<Integer>();

        int totalCost = 0;
        int numberOfLinks = network.getLinks().length;
        int numberOfNodes = 0;
        int from;
        int to;

        for(int i = 0; i < numberOfLinks; i++)
        {
            from = network.getLinks()[i].getFromNeuron();
            to = network.getLinks()[i].getToNeuron();

            if(!networkNodes.contains(from))
            {
                numberOfNodes++;
                networkNodes.add(from);
            }

            if(!networkNodes.contains(to))
            {
                numberOfNodes++;
                networkNodes.add(to);
            }
        }

        totalCost = numberOfLinks + numberOfNodes;

        return totalCost;
    }

    @Override
    public double calculateScore(MLMethod method)
    {
        NEATNetwork network = (NEATNetwork) method;
        RobotFactory robotFactory = new HomogeneousRobotFactory(getPhenotypeForNetwork(network),
                simConfig.getRobotMass(), simConfig.getRobotRadius(), simConfig.getRobotColour(),
                simConfig.getObjectsRobots(), simConfig.getRobotSensorEnergyCosts(),
                simConfig.getRobotNeuralEnergyCosts(), getNeuralEnergyCost(method));

        // Create the simulation and run it
        Simulation simulation = new Simulation(simConfig, robotFactory);
        simulation.setStopOnceCollected(true);
        double fitness = 0;
        double peformance =0;
        for (int i = 0; i < trialsPerIndividual; i++)
        {
            simulation.run();
            fitness += simulation.getFitness().getTeamFitness();
            fitness += 10 * (1.0 - simulation.getProgressFraction()); // Time bonus --> progress fraction is ratio of timesteps that were needed to gather all resources with respect to
                                                                      // total possible timesteps for each individual's trial. So if an individual doesn't gather all resources, progress fraction
                                                                      // will just land up being 1, and thus the time bonus will be 10*(1-1) = 0.
            peformance +=simulation.getProgressFraction();
        }

        // Get the fitness and update the total score
        double score = fitness / trialsPerIndividual;
        scoreStats.addValue(score);

        double score2 = calculateScore2(method);

        if (isEvolvingMorphology() || hyperNEATM)
        {
            sensorStats.addValue(score2);
        }

        double score3 = calculateScore3(method);
        neuralStats.addValue(score3);



        TimeTakenStats.addValue(peformance/trialsPerIndividual);

        SensorMorphology sensors = ((NEATMNetwork) network).getSensorMorphology();
        for(int i =0;i<sensors.getNumSensors();i++){
            AgentSensor sensor = sensors.getSensor(i);
            if(!sensor.getClass().getSimpleName().equals("BottomProximitySensor")){
                sensorParam.get(sensor.getClass().getSimpleName())[0].addValue(sensor.getFieldOfView());
                sensorParam.get(sensor.getClass().getSimpleName())[1].addValue(sensor.getBearing());
                sensorParam.get(sensor.getClass().getSimpleName())[2].addValue(sensor.getRange());
            }

        }

        DecimalFormat df = new DecimalFormat("0.000");
        df.setRoundingMode(RoundingMode.CEILING);


        log.debug("task performance: "+df.format(score)+"\tsensor complexity: "+df.format(score2)+"\tneural complexity: "+df.format(score3));

        return score;
    }

    public double calculateScore2(MLMethod method)  // second fitness function score (multi-objective) --> morphological complexity
    {
        NEATMNetwork network = (NEATMNetwork) method;
        double score = 100;
        for(int i =0;i< network.getSensorMorphology().getNumSensors();i++){
            AgentSensor sensor =network.getSensorMorphology().getSensor(i);

            SensorType type=convertNameToType(sensor.getClass().getSimpleName());
            if(type ==SensorType.BOTTOM_PROXIMITY){
                score =score -10;
            }
            else {
                //look at scores range and remove
                double removeRangePercentage = calculateFractionWithinRange(type.getDefaultSpecSet().getParameterSpec(ParameterType.RANGE).getRange(), sensor.getRange());
                score -= (5 * removeRangePercentage);

                //look at scores fov and remove
                double removeFOVPercentage = calculateFractionWithinRange(type.getDefaultSpecSet().getParameterSpec(ParameterType.FIELD_OF_VIEW).getRange(), sensor.getFieldOfView());
                score -= (5 * removeRangePercentage);
            }


        }
        return score;

    }

    public double calculateScore3(MLMethod method)  // neural complexity
    {
        NEATMNetwork network = (NEATMNetwork) method;
        //System.out.println(network.getLinks().length);
       // System.out.println((network.getLinks().length/100.0));
        return 100-(network.getLinks().length/40.0)*100;
    }



    public double calculateFractionWithinRange(Range range,float value){
        return (value -(range.min))/(range.max -(range.min));

    }

    public SensorType convertNameToType ( String SimpleName){

        if(SimpleName.equals("LowResCameraSensor")){
            return SensorType.LOW_RES_CAM;
        }
        else if(SimpleName.equals("LowResCameraSensor")){
            return SensorType.LOW_RES_CAM;
        }
        else if(SimpleName.equals("ColourProximitySensor")){
            return SensorType.COLOUR_PROXIMITY;
        }
        else if(SimpleName.equals("UltrasonicSensor")){
            return SensorType.ULTRASONIC;
        }
        else{
            return SensorType.BOTTOM_PROXIMITY;
        }



    }

    public void demo(MLMethod method)
    {
        // Create the robot and resource factories
        NEATNetwork network = (NEATNetwork) method;
        RobotFactory robotFactory = new HomogeneousRobotFactory(getPhenotypeForNetwork(network),
                simConfig.getRobotMass(), simConfig.getRobotRadius(), simConfig.getRobotColour(),
                simConfig.getObjectsRobots(), simConfig.getRobotSensorEnergyCosts(),
                simConfig.getRobotNeuralEnergyCosts(), getNeuralEnergyCost(method));

        // Create the simulation and run it
        Simulation simulation = new Simulation(simConfig, robotFactory);

        SimulationGUI video = new SimulationGUI(simulation);

        //new console which displays this simulation
        Console console = new Console(video);
        console.setVisible(true);
    }

    private Phenotype getPhenotypeForNetwork(NEATNetwork network)
    {
        if (isEvolvingMorphology() )
        {
            return new NEATMPhenotype((NEATMNetwork) network);
        }
        else
        {
            return new NEATPhenotype(network, sensorMorphology);
        }
    }

    public boolean isEvolvingMorphology()
    {
        return sensorMorphology == null;
    }

    public boolean isHyperNEATM()
    {
        return hyperNEATM;
    }

    public DescriptiveStatistics getTimeTakenStatistics()
    {
        return TimeTakenStats;
    }

    public DescriptiveStatistics getScoreStatistics()
    {
        return scoreStats;
    }

    public DescriptiveStatistics getSensorStatistics()
    {
        return sensorStats;
    }

    public DescriptiveStatistics getNeuralStatistics() { return neuralStats; }

    public HashMap<String,DescriptiveStatistics[]> getParamSensor(){
        return sensorParam;
    }


    @Override
    public boolean shouldMinimize()
    {
        return false;
    }

    @Override
    public boolean requireSingleThreaded()
    {
        return false;
    }

}
