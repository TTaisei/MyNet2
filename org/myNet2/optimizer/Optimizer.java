package org.myNet2.optimizer;

import java.util.ArrayList;
import java.util.Random;

import org.myNet2.network.*;
import org.myNet2.lossFunc.*;
import org.myNet2.layer.*;
import org.myNet2.*;

/**
 * Class for optimizer.
 */
public class Optimizer {
    public Random rand;
    /** Optimizing network */
    public Network net;
    /** Loss function of this network. */
    public LossFunction lossFunc;
    /** Length of this network. */
    public int layersLength;
    /** Learning rate. */
    public double eta = 0.01;

    /**
     * Constructor for this class.
     */
    protected Optimizer(){
        ;
    }

    /**
     * Set this random instance.
     */
    public void setRandom(){
        this.rand = new Random(0);
    }

    /**
     * Set this random instance.
     * @param seed Number of seed.
     */
    public void setRandom(long seed){
        this.rand = new Random(seed);
    }

    /**
     * Make data for mini batch learning.
     * @param x Input data.
     * @param t Answer.
     * @param batchSize Number of batch size.
     * @param rand Random instance.
     * @return Splited input data and answer.
     */
    protected Matrix[][] makeMiniBatch(Matrix x, Matrix t, int batchSize, Random rand){
        int rtnSize = x.row % batchSize == 0 ? x.row / batchSize : x.row / batchSize + 1;;
        int num;
        int i;
        ArrayList<Integer> order = new ArrayList<Integer>(rtnSize);
        ArrayList<Integer> check = new ArrayList<Integer>(rtnSize);

        for (i = 0; i < x.row; i++){
            check.add(i);
        }
        for (i = 0; i < x.row; i++){
            num = rand.nextInt(x.row - order.size());
            order.add(check.get(num));
            check.remove(num);
        }
        for (; i < rtnSize*batchSize; i++){
            order.add(rand.nextInt(x.row));
        }

        Matrix x_ = x.vsort(order);
        Matrix t_ = t.vsort(order);
        return new Matrix[][]{x_.vsplit(rtnSize), t_.vsplit(rtnSize)};
    }
}
