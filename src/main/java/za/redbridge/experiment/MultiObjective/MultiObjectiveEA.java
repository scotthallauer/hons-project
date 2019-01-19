package za.redbridge.experiment.MultiObjective;

import org.encog.Encog;
import org.encog.EncogError;
import org.encog.EncogShutdownTask;
import org.encog.mathutil.randomize.factory.RandomFactory;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLContext;
import org.encog.ml.MLMethod;
import org.encog.ml.ea.codec.GeneticCODEC;
import org.encog.ml.ea.codec.GenomeAsPhenomeCODEC;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.opp.EvolutionaryOperator;
import org.encog.ml.ea.opp.OperationList;
import org.encog.ml.ea.opp.selection.SelectionOperator;
import org.encog.ml.ea.opp.selection.TournamentSelection;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.rules.BasicRuleHolder;
import org.encog.ml.ea.rules.RuleHolder;
import org.encog.ml.ea.score.AdjustScore;
import org.encog.ml.ea.sort.*;
import org.encog.ml.ea.species.SingleSpeciation;
import org.encog.ml.ea.species.Speciation;
import org.encog.ml.ea.species.Species;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.ml.genetic.GeneticError;
import org.encog.util.concurrency.MultiThreadable;
import org.encog.util.logging.EncogLogging;
import za.redbridge.experiment.MultiObjective.Comparator.DistanceComparator;
import za.redbridge.experiment.MultiObjective.Comparator.MaximisingObjectiveComparator;
import za.redbridge.experiment.MultiObjective.Comparator.ScoreComparator;
import za.redbridge.experiment.ScoreCalculator;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MultiObjectiveEA implements EvolutionaryAlgorithm, MultiThreadable,
        EncogShutdownTask, Serializable
{

    /**
     * Random number factory.
     */
    private RandomFactory randomNumberFactory = Encog.getInstance()
            .getRandomFactory().factorFactory();

    /**
     * The thread pool executor.
     */
    private ExecutorService taskExecutor;

    /**
     * The number of times to try certian operations, so an endless loop does
     * not occur.
     */
    private int maxTries = 5;

    /**
     * Should exceptions be ignored.
     */
    private boolean ignoreExceptions;

    /**
     * The validation mode.
     */
    private boolean validationMode;

    /**
     * The selection operator.
     */
    private SelectionOperator selection;

    /**
     * Holds rewrite and constraint rules.
     */
    private RuleHolder rules;

    /**
     * The speciation method.
     */
    private Speciation speciation = new SingleSpeciation();

    /**
     * The iteration number.
     */
    private int iteration;

    /**
     * The genome comparator.
     */
    private GenomeComparator selectionComparator;

    /**
     * The best ever genome.
     */
    private Genome bestGenome;

    /**
     * The genome comparator.
     */
    private GenomeComparator bestComparator;

    /**
     * The operators. to use.
     */
    private final OperationList operators = new OperationList();


    /**
     * The number of requested threads.
     */
    private int threadCount;

    /**
     * The mutation to be used on the top genome. We want to only modify its
     * weights.
     */
    private EvolutionaryOperator champMutation;

    /**
     * The CODEC to use to convert between genome and phenome.
     */
    private GeneticCODEC codec = new GenomeAsPhenomeCODEC();

    /**
     * The population.
     */
    private Population population;

    /**
     * The score calculation function.
     */
    private final CalculateScore scoreFunction;

    /**
     * The population for the next iteration.
     */
    private final List<Genome> newPopulation = new ArrayList<Genome>();

    /**
     * The actual thread count.
     */
    private int actualThreadCount = -1;

    /**
     * Holds the threads used each iteration.
     */
    private final List<Callable<Object>> threadList = new ArrayList<Callable<Object>>();

    /**
     * This property stores any error that might be reported by a thread.
     */
    private Throwable reportedError;

    private int maxOperationErrors = 500;

    /**
      * This variable stores the first pareto front for each generation (used by StatsRecorder)
     */
    ArrayList<MultiObjectiveGenome> paretoFront0 = new ArrayList<>();

    /**
     * This variable stores the second pareto front for each generation (used by StatsRecorder)
     */
    ArrayList<MultiObjectiveGenome> paretoFront1 = new ArrayList<>();


    /**
     * Create a trainer for a score function.
     * @param thePopulation The population.
     * @param theScoreFunction The score function.
     */
    public MultiObjectiveEA(Population thePopulation, CalculateScore theScoreFunction)
    {
        this.population = thePopulation;
        this.scoreFunction = theScoreFunction;
        this.selection = new TournamentSelection(this, 4);
        this.rules = new BasicRuleHolder();


        // set the score compare method
        if (theScoreFunction.shouldMinimize())
        {
            this.selectionComparator = new MinimizeAdjustedScoreComp();
            this.bestComparator = new MinimizeScoreComp();
        } else
        {
            this.selectionComparator = new MaximizeAdjustedScoreComp();
            this.bestComparator = new MaximizeScoreComp();
        }

        // set the iteration
        for (final Species species : thePopulation.getSpecies())
        {
            for (final Genome genome : species.getMembers())
            {
                setIteration(Math.max(getIteration(),
                        genome.getBirthGeneration()));
            }
        }

        // Set a best genome, just so it is not null.
        // We won't know the true best genome until the first iteration.
        if( this.population.getSpecies().size()>0 && this.population.getSpecies().get(0).getMembers().size()>0 ) {
            this.bestGenome = this.population.getSpecies().get(0).getMembers().get(0);
        }
    }


    /**
     * The first iteration of Multi-Objective NEAT. Also determines number of threads to use.
     */


    public void firstIterationResume(){



        getSpeciation().init(this);

        // Threads - find out how many threads to use
        if (this.getThreadCount() == 0) this.actualThreadCount = Runtime.getRuntime().availableProcessors();
        else this.actualThreadCount = this.getThreadCount();


        // start up the thread pool
        if (this.actualThreadCount == 1)
        {
            this.taskExecutor = Executors.newSingleThreadScheduledExecutor();
        }
        else
        {
            this.taskExecutor = Executors.newFixedThreadPool(this.actualThreadCount);
        }

        // register for shutdown
        Encog.getInstance().addShutdownTask(this);

        // just pick the first genome with a valid score as best, it will be
        // updated later.
        // also most populations are sorted this way after training finishes
        // (for reload)
        // if there is an empty population, the constructor would have blow
        final List<Genome> list = getPopulation().flatten();

        int idx = 0;
        do
        {
            this.bestGenome = list.get(idx++);
        } while (idx < list.size()
                && (Double.isInfinite(this.bestGenome.getScore()) || Double
                .isNaN(this.bestGenome.getScore())));

        getPopulation().setBestGenome(this.bestGenome);


        //we dont wanna respeciate but rather redefine the final shares for the next iteration

        double totalSpeciesScore = 0;
        for (final Species species : population.getSpecies()) {
            totalSpeciesScore += species.calculateShare(getScoreFunction().shouldMinimize(), population.getPopulationSize()*2);
        }

        if (population.getSpecies().size() == 0) {
            throw new EncogError("Can't speciate, there are no species.2");
        }
        if (totalSpeciesScore < Encog.DEFAULT_DOUBLE_EQUAL) {
            // This should not happen much, or if it does, only in the
            // beginning.
            // All species scored zero. So they are all equally bad. Just divide
            // up the right to produce offspring evenly.
            divideEven(population.getSpecies());
        } else {
            // Divide up the number of offspring produced to the most fit
            // species.
            divideByFittestSpecies(population.getSpecies(), totalSpeciesScore);
        }

        levelOff();



    }

    public void firstIteration()
    {
        // init speciation
        getSpeciation().init(this);

        // Threads - find out how many threads to use
        if (this.getThreadCount() == 0) this.actualThreadCount = Runtime.getRuntime().availableProcessors();
        else this.actualThreadCount = this.getThreadCount();

        // Score the initial (parent) population
        final MultiObjectiveParallelScore pscore = new MultiObjectiveParallelScore(getPopulation(),
                getCODEC(), new ArrayList<AdjustScore>(), getScoreFunction(),
                this.actualThreadCount);
        pscore.setThreadCount(this.actualThreadCount);
        pscore.process();
        this.actualThreadCount = pscore.getThreadCount();

        // start up the thread pool
        if (this.actualThreadCount == 1)
        {
            this.taskExecutor = Executors.newSingleThreadScheduledExecutor();
        }
        else
        {
            this.taskExecutor = Executors.newFixedThreadPool(this.actualThreadCount);
        }

        // register for shutdown
        Encog.getInstance().addShutdownTask(this);

        // just pick the first genome with a valid score as best, it will be
        // updated later.
        // also most populations are sorted this way after training finishes
        // (for reload)
        // if there is an empty population, the constructor would have blow
        final List<Genome> list = getPopulation().flatten();

        int idx = 0;
        do
        {
            this.bestGenome = list.get(idx++);
        } while (idx < list.size()
                && (Double.isInfinite(this.bestGenome.getScore()) || Double
                .isNaN(this.bestGenome.getScore())));

        getPopulation().setBestGenome(this.bestGenome);

        // speciate
        final List<Genome> genomes = getPopulation().flatten();
        this.speciation.performSpeciation(genomes);

        // purge invalid genomes
        this.population.purgeInvalidGenomes();

        // Set Non-Domination ranks and Crowding Distance scores
        setNonDominationAndCrowdDists(true);


        // Set final scores --> based on sort index (maybe do in method setNonDomAndCRowdDsi

        // ensure all species are chosen otherwise may have a problem with choosing N members
        // but this will rarely ever happen in first generation


        // done! (parentPopulation set)
    }

    /**
     * Perform a training iteration. Also called generations or epochs.
     */
    @Override
    public void iteration()
    {
        if (this.actualThreadCount == -1) {
            firstIteration();
        }
        if (getPopulation().getSpecies().size() == 0) {
            throw new EncogError("Population is empty, there are no species.");
        }

        this.iteration++;


        // Create offspringPopulation (N)
        // offspringPopulation = mutateAndCrossover(parentPopulation)

        this.newPopulation.clear();
        //todo: look into old best Genome (shouldnt have one because we are combining population)

        // execute species in parallel
        this.threadList.clear();

        for (final Species species : getPopulation().getSpecies()) {
            int numToSpawn = species.getOffspringCount();
            for(Genome g: species.getMembers()){
                addChild(g);
            }

            // now add one task for each offspring that each species is allowed

            while (numToSpawn-- > 0) {
                final MultiObjectiveEAWorker worker = new MultiObjectiveEAWorker(this, species);
                this.threadList.add(worker);
            }
        }
        

        // run all threads and wait for them to finish
        // each thread creates one bae! the bae is then evaluated
        try {
            this.taskExecutor.invokeAll(this.threadList);
        } catch (final InterruptedException e) {
            EncogLogging.log(e);
        }

        // handle any errors that might have happened in the threads
        if (this.reportedError != null && !getShouldIgnoreExceptions()) {
            throw new GeneticError(this.reportedError);
        }
        //add this point --> new population is actually combined population


        // speciate(combinedPopulation) --> In Encog this is added to the actual population

        this.speciation.performSpeciation(this.newPopulation);

        this.population.purgeInvalidGenomes();  // delete genomes with NaN scores

        //call method on combined population (nonDomCrowd Sort

            // Selection Phase 1:

            // nonDominationSort(combinedPopulation)

            // setCrowdingDistances(combinedPopulation)

            // we now have a way of ranking species in combinedPopulation

            // now select the best species from combinedPopulation for Phase 2
            // setOfBestSpecies = selectBestSpecies(combinedPopulation)

            // Selection Phase 2:

            // Choose N best individuals from the set of best species
            // NChosenIndividuals = serialSpeciesProgression(setOfBestSpecies)

            // Set parent population to these N selected individuals
            // parentPopulation = the N selected individuals

        setNonDominationAndCrowdDists(false);

    }

    private void setNonDominationAndCrowdDists(boolean isFirstGen)
    {
        HashMap<MultiObjectiveGenome, ArrayList<MultiObjectiveGenome>> S = new HashMap<>();
        HashMap<MultiObjectiveGenome, Integer> n = new HashMap<>();  // number of individuals that dominate p
        ArrayList<ArrayList<MultiObjectiveGenome>> Fronts = new ArrayList<>();
        ArrayList<Species> SelectedSpecies = new ArrayList<Species>();
        int scoreCount = population.getPopulationSize()*2;              // size 2N since selection pool contains N parents and N children (elitism!)
        int UpperBoundSpecies = population.getPopulationSize()/2;       //populationsize/objectiveSize

        Fronts.add(new ArrayList<MultiObjectiveGenome>());
        for(Species species : population.getSpecies())
        {
            for(Genome pp : species.getMembers())
            {

                MultiObjectiveGenome p = (MultiObjectiveGenome)pp;
                if(pp.getSpecies()==null)  // When a new species is created, the BasicSpecies class forgets to set the species variable of the leader genome in that species.
                {
                    pp.setSpecies(species);
                }
                S.put(p, new ArrayList<MultiObjectiveGenome>());
                n.put(p, 0);
                for(Species species2 : population.getSpecies())
                {
                    for(Genome qq : species2.getMembers())
                    {
                        MultiObjectiveGenome q = (MultiObjectiveGenome)qq;
                        if(checkADominatesB(p,q))
                        {
                            S.get(p).add(q);
                        }
                        else if (checkADominatesB(q,p))
                        {
                            n.put(p,n.get(p)+1);
                        }
                    }
                }
                if(n.get(p).equals(0))   // if p dominated by no one
                {
                    p.setRank(0);    // then p belongs to the 1st Pareto front
                    Fronts.get(0).add(p);   // add p to first front
                }
            }
        }


        // First iteration now finished - everyone scored


        int i = 0; // initialise Front counter to 0

        while(!(Fronts.get(i).size()==0))
        {
            Fronts.set(i,crowdingDistance(Fronts.get(i))); //sorts on crowding distance
            ArrayList<MultiObjectiveGenome> Q = new ArrayList<>();  // will store individuals in Fronts[i+1]

            for(MultiObjectiveGenome p: Fronts.get(i))
            {
                //Selected Species add
                p.setScore(scoreCount);
                p.setAdjustedScore(scoreCount);
                scoreCount--;
                if(!SelectedSpecies.contains(p.getSpecies())) {
                    if(SelectedSpecies.size()< UpperBoundSpecies)
                    {
                        SelectedSpecies.add(p.getSpecies());
                    }
                }

                for(MultiObjectiveGenome q: S.get(p))   // for each invidiual q that is dominated by p
                {
                    n.put(q,n.get(q)-1);             // decrement # of individuals that dominate q
                    if(n.get(q).equals(0))              // if q is only dominated by the first front
                    {
                        q.setRank(i+1);                 // then it belongs in the second front
                        Q.add(q);
                    }
                }
            }

            Fronts.add(Q); // update the next front to be Q

            i++;                                     // increment front counter
        }

        // update 1st and 2nd pareto front lists used by StatsRecorder
        paretoFront0 = new ArrayList<>();
        paretoFront0.addAll(Fronts.get(0));
        if(Fronts.size() > 1)
        {
            paretoFront1 = new ArrayList<>();
            paretoFront1.addAll(Fronts.get(1));
        }

        if(isFirstGen){
            return;
        }

        // Sort each species
        for(int j =0;j<SelectedSpecies.size();j++)
        {
           Collections.sort(SelectedSpecies.get(j).getMembers(), new ScoreComparator<Genome>());
        }


        // Serial Progression to select N individuals from the list of sorted species we selected in phase 1.
        //---------------------

        int numSelectedSoFar = 0;
        int row = 0;
        boolean done = false;
        while(!done)
        {
            done = true;
            for (int col = 0; col < SelectedSpecies.size(); col++)
            {
                if (row < SelectedSpecies.get(col).getMembers().size())
                {
                    done = false;
                    if (numSelectedSoFar >= population.getPopulationSize())
                    {
                        SelectedSpecies.get(col).getMembers().subList(row,SelectedSpecies.get(col).getMembers().size()).clear();
                    }
                    else
                    {
                        numSelectedSoFar++;
                    }
                }
            }
            row++;
        }

        double totalSpeciesScore = 0;
        for (final Species species : SelectedSpecies)
        {
            totalSpeciesScore += species.calculateShare(false, population.getPopulationSize()*2);
        }
        divideByFittestSpecies(SelectedSpecies,totalSpeciesScore);

        population.getSpecies().clear();
        population.getSpecies().addAll(SelectedSpecies);

    }

    private ArrayList<MultiObjectiveGenome> crowdingDistance(ArrayList<MultiObjectiveGenome> front){
        int l = front.size();
        for(MultiObjectiveGenome i : front)
        {
            i.setDistance(0.0);
        }

        MaximisingObjectiveComparator<MultiObjectiveGenome> comparator = new MaximisingObjectiveComparator<>();
       // for(int i =0;i<2;i++){ //change for MO!!!!
        for(int i =0;i<3;i=i+2){
            comparator.setObjectiveIndex(i);
            Collections.sort(front,comparator);
            front.get(0).setDistance(Double.POSITIVE_INFINITY);
            front.get(l-1).setDistance(Double.POSITIVE_INFINITY);
            for(int j =1;j<l-1;j++){
                front.get(j).setDistance(front.get(j).getDistance()+(front.get(j+1).getScoreVector().get(i)-front.get(j-1).getScoreVector().get(i)));
            }

        }

        Collections.sort(front, new DistanceComparator<MultiObjectiveGenome>());
        return front;



    }
    //
    private boolean checkADominatesB(MultiObjectiveGenome a, MultiObjectiveGenome b) // check if a dominates b
    {
        int secondScore =2; //for Morph - change to 1
        if(a.getScoreVector().get(0) >= b.getScoreVector().get(0) && a.getScoreVector().get(secondScore) >= b.getScoreVector().get(secondScore))
        {
            if(a.getScoreVector().get(0) > b.getScoreVector().get(0) || a.getScoreVector().get(secondScore) > b.getScoreVector().get(secondScore))
            {
                return true; // a dominates b
            }
        }
        return false;
    }

    /**
     * @return the maxOperationErrors
     */
    public int getMaxOperationErrors() {
        return maxOperationErrors;
    }

    /**
     * @param maxOperationErrors the maxOperationErrors to set
     */
    public void setMaxOperationErrors(int maxOperationErrors) {
        this.maxOperationErrors = maxOperationErrors;
    }


    /**
     * Called by a thread to report an error.
     *
     * @param t
     *            The error reported.
     */
    public void reportError(final Throwable t) {
        synchronized (this) {
            if (this.reportedError == null) {
                this.reportedError = t;
            }
        }
    }


    /**
     * Add a child to the next iteration.
     *
     * @param genome
     *            The child.
     * @return True, if the child was added successfully.
     */
    public boolean addChild(final Genome genome)
    {
        synchronized (this.newPopulation)
        {
            if (this.newPopulation.size() < getPopulation().getPopulationSize()*2)
            {

                if (isValidationMode())
                {
                    if (this.newPopulation.contains(genome))
                    {
                        throw new EncogError(
                                "Genome already added to population: "
                                        + genome.toString());
                    }
                }

                this.newPopulation.add(genome);

                if (!Double.isInfinite(genome.getScore())
                        && !Double.isNaN(genome.getScore())
                        && getBestComparator().isBetterThan(genome,
                        this.bestGenome))
                {
                    this.bestGenome = genome;
                    getPopulation().setBestGenome(this.bestGenome);
                }
                return true;
            }
            else
            {
                System.out.println("tooo big");
                return false;
            }
        }
    }

    @Override
    public void performShutdownTask()
    {
        finishTraining();
    }

    /**
     * Add an operation.
     *
     * @param probability The probability of using this operator.
     * @param opp
     */
    @Override
    public void addOperation(final double probability, final EvolutionaryOperator opp)
    {
        getOperators().add(probability, opp);
        opp.init(this);
    }


    /**
     * @return the randomNumberFactory
     */
    public RandomFactory getRandomNumberFactory() {
        return this.randomNumberFactory;
    }


    /**
     * Add a score adjuster. Score adjusters are used to adjust the adjusted
     * score of a genome. This allows bonuses and penalties to be applied for
     * desirable or undesirable traits.
     * desirable or undesirable trits.
     *
     * @param scoreAdjust The score adjustor to add.
     */
    @Override
    public void addScoreAdjuster(AdjustScore scoreAdjust)
    {

    }

    /**
     * Calculate the score for a genome.
     *
     * @param g The genome to calculate the score for.
     */
    @Override
    public void calculateScore(Genome g)
    {
        // try rewrite
        this.rules.rewrite(g);

        // decode
        final MLMethod phenotype = getCODEC().decode(g);
        double score;
        double score2;
        double score3;

        // deal with invalid decode
        if (phenotype == null)
        {
            if (getBestComparator().shouldMinimize())
            {
                score = Double.POSITIVE_INFINITY;
                score2 = Double.POSITIVE_INFINITY;
                score3 = Double.POSITIVE_INFINITY;
            }
            else
            {
                score = Double.NEGATIVE_INFINITY;
                score2 = Double.NEGATIVE_INFINITY;
                score3 = Double.NEGATIVE_INFINITY;
            }
        }
        else
        {
            if (phenotype instanceof MLContext)
            {
                ((MLContext) phenotype).clearContext();
            }
            score = getScoreFunction().calculateScore(phenotype);   // score
            score2 = ((ScoreCalculator) getScoreFunction()).calculateScore2(phenotype); // morphology simplicity
            score3 = ((ScoreCalculator) getScoreFunction()).calculateScore3(phenotype); // neural complexity
        }

        // now set the scores
        ( (MultiObjectiveGenome) g).setScore(0,score);
        ( (MultiObjectiveGenome) g).setScore(1,score2);
        ( (MultiObjectiveGenome) g).setScore(2,score3);

        g.setScore(score);
        g.setAdjustedScore(score);
    }

    /**
     * Called when training is finished. This allows the EA to properly shut
     * down.
     */
    @Override
    public void finishTraining()
    {
        // wait for threadpool to shutdown
        if (this.taskExecutor != null) {
            this.taskExecutor.shutdown();
            try {
                this.taskExecutor.awaitTermination(Long.MAX_VALUE,
                        TimeUnit.MINUTES);
            } catch (final InterruptedException e) {
                throw new GeneticError(e);
            } finally {
                this.taskExecutor = null;
                Encog.getInstance().removeShutdownTask(this);
            }
        }
    }

    /**
     * Get the comparator that is used to choose the "true best" genome. This
     * uses the real score, and not the adjusted score.
     *
     * @return The best comparator.
     */
    @Override
    public GenomeComparator getBestComparator()
    {
        return this.bestComparator;
    }

    /**
     * @return The current best genome. This genome is safe to use while the EA
     * is running. Genomes are not modified. They simply produce
     * "offspring".
     */
    @Override
    public Genome getBestGenome() { return this.bestGenome;}

    /**
     * @return The CODEC that is used to transform between genome and phenome.
     */
    @Override
    public GeneticCODEC getCODEC()
    {
        return this.codec;
    }

    /**
     * @return The current score. This value should either be minimized or
     * maximized, depending on the score function.
     */
    @Override
    public double getError()
    {
        // do we have a best genome, and does it have an error?
        if (this.bestGenome != null)
        {
            double err = this.bestGenome.getScore();
            if( !Double.isNaN(err) )
            {
                return err;
            }
        }
        // otherwise, assume the worst!
        if (getScoreFunction().shouldMinimize())
        {
            return Double.POSITIVE_INFINITY;
        }
        else
        {
            return Double.NEGATIVE_INFINITY;
        }
    }

    /**
     * @return The current iteration number. Also sometimes referred to as
     * generation or epoch.
     */
    @Override
    public int getIteration()
    {
        return this.iteration;
    }

    /**
     * @return The maximum size an individual genome can be. This is an
     * arbitrary number defined by the genome. Lower numbers are less
     * complex.
     */
    @Override
    public int getMaxIndividualSize()
    {
        return this.population.getMaxIndividualSize();
    }

    /**
     * @return The maximum number to try certain genetic operations. This
     * prevents endless loops.
     */
    @Override
    public int getMaxTries()
    {
        return this.maxTries;
    }

    /**
     * @return The operators.
     */
    @Override
    public OperationList getOperators()
    {
        return this.operators;
    }

    /**
     * @return The population.
     */
    @Override
    public Population getPopulation()
    {
        return this.population;
    }

    /**
     * @return The rules holder, contains rewrite and constraint rules.
     */
    @Override
    public RuleHolder getRules()
    {
        return this.rules;
    }

    /**
     * @return The score adjusters. This allows bonuses and penalties to be
     * applied for desirable or undesirable traits.
     */
    @Override
    public List<AdjustScore> getScoreAdjusters()
    {
        return null;
    }   // todo: check if this is needed

    /**
     * @return The score function.
     */
    @Override
    public CalculateScore getScoreFunction()
    {
        return this.scoreFunction;
    }

    /**
     * @return The selection operator. Used to choose genomes.
     */
    @Override
    public SelectionOperator getSelection()
    {
        return this.selection;
    }

    /**
     * Get the comparator that is used to choose the "best" genome for
     * selection, as opposed to the "true best". This uses the adjusted score,
     * and not the score.
     *
     * @return The selection comparator.
     */
    @Override
    public GenomeComparator getSelectionComparator()
    {
        return this.selectionComparator;
    }

    /**
     * @return True if exceptions that occur during genetic operations should be
     * ignored.
     */
    @Override
    public boolean getShouldIgnoreExceptions()
    {
        return this.ignoreExceptions;
    }

    /**
     * @return The speciation method.
     */
    @Override
    public Speciation getSpeciation()
    {
        return this.speciation;
    }

    /**
     * @return True if any genome validators should be applied.
     */
    @Override
    public boolean isValidationMode()
    {
        return this.validationMode;
    }



    /**
     * Set the comparator that is used to choose the "true best" genome. This
     * uses the real score, and not the adjusted score.
     *
     * @param bestComparator The best comparator.
     */
    @Override
    public void setBestComparator(final GenomeComparator bestComparator)
    {
        this.bestComparator = bestComparator;
    }

    /**
     * Set the population.
     *
     * @param thePopulation The population.
     */
    @Override
    public void setPopulation(final Population thePopulation)
    {
        this.population = population;
    }

    /**
     * Set the rules holder to use.
     *
     * @param rules The rules holder.
     */
    @Override
    public void setRules(final RuleHolder rules)
    {
        this.rules = rules;
    }

    /**
     * Set the selection operator.
     *
     * @param selection The selection operator.
     */
    @Override
    public void setSelection(final SelectionOperator selection)
    {
        this.selection = selection;
    }

    /**
     * Set the comparator that is used to choose the "best" genome for
     * selection, as opposed to the "true best". This uses the adjusted score,
     * and not the score.
     *
     * @param selectionComparator The selection comparator.
     */
    @Override
    public void setSelectionComparator(final GenomeComparator selectionComparator)
    {
        this.selectionComparator = selectionComparator;
    }

    /**
     * Determines if genetic operator exceptions should be ignored.
     *
     * @param b True if exceptions should be ignored.
     */
    @Override
    public void setShouldIgnoreExceptions(final boolean b)
    {
        this.ignoreExceptions = b;
    }

    /**
     * Set the speciation method.
     *
     * @param speciation The speciation method.
     */
    @Override
    public void setSpeciation(final Speciation speciation)
    {
        this.speciation = speciation;
    }

    /**
     * Determine if the genomes should be validated. This takes more time but
     * can help isolate a problem.
     *
     * @param validationMode True, if validation mode is enabled.
     */
    @Override
    public void setValidationMode(final boolean validationMode)
    {
        this.validationMode = validationMode;
    }

    /**
     * @return The number of threads to use, 0 to automatically
     * determine based on core count.
     */
    @Override
    public int getThreadCount()
    {
        return this.threadCount;
    }

    /**
     * Set the number of threads to use.
     *
     * @param numThreads The number of threads to use, or zero to
     *                   automatically determine based on core count.
     */
    @Override
    public void setThreadCount(final int numThreads)
    {
        this.threadCount = numThreads;
    }

    /**
     * Set the CODEC to use.
     *
     * @param theCodec
     *            The CODEC to use.
     */
    public void setCODEC(final GeneticCODEC theCodec)
    {
        this.codec = theCodec;
    }

    /**
     * @param champMutation
     *            the champMutation to set
     */
    public void setChampMutation(final EvolutionaryOperator champMutation) {
        this.champMutation = champMutation;
    }

    /**
     * Set the current iteration number.
     *
     * @param iteration
     *            The iteration number.
     */
    public void setIteration(final int iteration)
    {
        this.iteration = iteration;
    }

    /**
     * Taken from ThresholdSpeciation because private
     * If no species has a good score then divide the potential offspring amount
     * all species evenly.
     *
     * @param speciesCollection
     *            The current set of species.
     */
    private void divideEven(final List<Species> speciesCollection) {
        final double ratio = 1.0 / speciesCollection.size();
        for (final Species species : speciesCollection) {
            final int share = (int) Math.round(ratio
                    *getPopulation().getPopulationSize());
            species.setOffspringCount(share);
        }
    }

    /**
     * Taken from ThresholdSpeciation because private
     * Divide up the potential offspring by the most fit species. To do this we
     * look at the total species score, vs each individual species percent
     * contribution to that score.
     *
     * @param speciesCollection
     *            The current species list.
     * @param totalSpeciesScore
     *            The total score over all species.
     */
    public void divideByFittestSpecies(final List<Species> speciesCollection,
                                        final double totalSpeciesScore) {
        // loop over all species and calculate its share
        final Object[] speciesArray = speciesCollection.toArray();
        for (final Object element : speciesArray) {
            final Species species = (Species) element;
            // calculate the species share based on the percent of the total
            // species score
            int share = (int) Math
                    .round((species.getOffspringShare() / totalSpeciesScore)
                            * getPopulation().getPopulationSize());

            // do not give the best species a zero-share
            if ((species == population.getBestGenome().getSpecies()) && (share == 0)) {
                share = 1;
            }

            /*
            // if the share is zero, then remove the species
            if ((species.getMembers().size() == 0) || (share == 0))
            {
                speciesCollection.remove(species);    // todo: was this
            }
            else
            {
                // otherwise assign a share and sort the members.
                species.setOffspringCount(share);
            }
            */
            species.setOffspringCount(share); // remove this if uncommenting above lines
        }
    }
    /**
     * Level off all of the species shares so that they add up to the desired
     * population size. If they do not add up to the desired species size, this
     * was a result of rounding the floating point share amounts to integers.
     */
    private void levelOff() {
        int total = 0;
        final List<Species> list = this.population.getSpecies();

        if (list.size() == 0) {
            throw new EncogError(
                    "Can't speciate, next generation contains no species.");
        }

        Collections.sort(list, new SpeciesComparator(this));

        // best species gets at least one offspring
        if (list.get(0).getOffspringCount() == 0) {
            list.get(0).setOffspringCount(1);
        }

        // total up offspring
        for (final Species species : list) {
            total += species.getOffspringCount();
        }

        // how does the total offspring count match the target
        int diff = this.population.getPopulationSize() - total;

        if (diff < 0) {
            // need less offspring
            int index = list.size() - 1;
            while ((diff != 0) && (index > 0)) {
                final Species species = list.get(index);
                final int t = Math.min(species.getOffspringCount(),
                        Math.abs(diff));
                species.setOffspringCount(species.getOffspringCount() - t);
                /*
                if (species.getOffspringCount() == 0)
                {
                    list.remove(index);                 // todo: related to
                }
                */
                diff += t;
                index--;
            }
        } else {
            // need more offspring
            list.get(0).setOffspringCount(
                    list.get(0).getOffspringCount() + diff);
        }
    }
    public ArrayList<MultiObjectiveGenome> getFirstParetoFront()
    {
        return paretoFront0;
    }

    public ArrayList<MultiObjectiveGenome> getSecondParetoFront()
    {
        return paretoFront1;
    }

}
