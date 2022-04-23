package org.MyNet2.optimizer;

import java.util.ArrayList;
import java.util.Random;
import java.io.PrintWriter;
import java.io.IOException;

import org.MyNet2.network.*;
import org.MyNet2.lossFunc.*;
import org.MyNet2.layer.*;
import org.MyNet2.*;

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
    // protected Optimizer(){
    public Optimizer(){
        ;
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
        int rtnSize = (int)(x.row / batchSize) + 1;
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

    /**
     * Doing back propagation.
     * @param x input matrix.
     * @param y Result of forward propagation.
     * @param t Answer.
     */
    protected void back(Matrix x, Matrix y, Matrix t){
        this.backLastLayer(x, y, t);
    }

    /**
     * Doing back propagation to last layer.
     * @param x input matrix.
     * @param y Result of forward propagation.
     * @param t Answer.
     */
    // protected backLastLayer(Matrix x, Matrix y, Matrix t){
    public void backLastLayer(Matrix x, Matrix y, Matrix t){
        Layer lastLayer = this.net.layers[this.layersLength-1];
        Layer preLayer = this.net.layers[this.layersLength-2];
        Matrix cal;

        cal = this.lossFunc.diff(lastLayer.a, t);
        System.out.println(cal);
        System.out.println(lastLayer.x);
        System.out.println(lastLayer.actFunc.diff(lastLayer.x));
        cal = cal.T().dot(lastLayer.actFunc.diff(lastLayer.x));
        System.out.println(cal);
        lastLayer.delta = cal.clone();
        // for (int i = 0; i < lastLyaer.nodesNum; i++){
        //     lastLayer.delta.matrix[i][0] = cal.matrix[i][0];
        // }
        cal = lastLayer.delta.dot(preLayer.a);
        lastLayer.w.add(cal.mult(-this.eta));
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param batchSize Size of batch.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, int batchSize){
        Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
        Matrix[] xs = xt[0];
        Matrix[] ts = xt[1];
        Matrix y = ts[0].clone();
        int backNum = (int)(x.row / batchSize) + 1;

        for (int i = 0; i < nEpoch; i++){
            System.out.printf("Epoch %d/%d\n", i+1, nEpoch);
            for (int j = 0; j < backNum; j++){
                y = this.net.forward(xs[j]);
                this.back(xs[j], y, ts[j]);
                System.out.printf("\rloss: %.4f", this.lossFunc.calc(y, ts[j]));
            }
            System.out.println();
        }

        return y;
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param batchSize Size of batch.
     * @param valX Input layer for validation.
     * @param valT Answer for validation.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, int batchSize,
                      Matrix valX, Matrix valT){
        Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
        Matrix[] xs = xt[0];
        Matrix[] ts = xt[1];
        Matrix[][] valxt = this.makeMiniBatch(valX, valT, batchSize, rand);
        Matrix[] valxs = valxt[0];
        Matrix[] valts = valxt[1];
        Matrix y = ts[0].clone();
        Matrix valY;
        int backNum = (int)(x.row / batchSize) + 1;

        for (int i = 0; i < nEpoch; i++){
            System.out.printf("Epoch %d/%d\n", i+1, nEpoch);
            for (int j = 0; j < backNum; j++){
                valY = this.net.forward(valxs[j]);
                y = this.net.forward(xs[j]);
                this.back(xs[j], y, ts[j]);
                System.out.printf(
                    "\rloss: %.4f - valLoss: %.4f",
                    this.lossFunc.calc(y, ts[j]),
                    this.lossFunc.calc(valY, valts[j])
                );
            }
            System.out.println();
        }

        return y;
    }

    /**
     * Run learning and save log.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param batchSize Size of batch.
     * @param fileName Name of logging file.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, int batchSize, String fileName){
        Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
        Matrix[] xs = xt[0];
        Matrix[] ts = xt[1];
        Matrix y = ts[0].clone();
        int backNum = (int)(x.row / batchSize) + 1;
        double loss = 0.;

        try(
            PrintWriter fp = new PrintWriter(fileName);
        ){
            fp.write("Epoch,loss\n");
            for (int i = 0; i < nEpoch; i++){
                for (int j = 0; j < backNum; j++){
                    y = this.net.forward(xs[j]);
                    this.back(xs[j], y, ts[j]);
                }
                loss = this.lossFunc.calc(y, ts[ts.length-1]);
                fp.printf("%d,%f\n", i+1, loss);
            }
        }catch (IOException e){
            System.out.println("IO Exception");
            System.exit(-1);
        }

        return y;
    }

    /**
     * Run learning and save log.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param batchSize Size of batch.
     * @param valX Input layer for validation.
     * @param valT Answer for validation.
     * @param fileName Name of logging file.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, int batchSize,
                      Matrix valX, Matrix valT, String fileName){
        Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
        Matrix[] xs = xt[0];
        Matrix[] ts = xt[1];
        Matrix y = ts[0].clone();
        Matrix valY;
        int backNum = (int)(x.row / batchSize) + 1;
        double loss = 0., valLoss = 0.;

        try(
            PrintWriter fp = new PrintWriter(fileName);
        ){
            fp.write("Epoch,loss,valLoss\n");
            for (int i = 0; i < nEpoch; i++){
                for (int j = 0; j < backNum; j++){
                    y = this.net.forward(xs[j]);
                    this.back(xs[j], y, ts[j]);
                }
                valY = this.net.forward(valX);
                loss = this.lossFunc.calc(y, ts[ts.length-1]);
                valLoss = this.lossFunc.calc(valY, valT);
                fp.printf("%d,%f,%f\n", i+1, loss, valLoss);
            }
        }catch (IOException e){
            System.out.println("IO Exception");
            System.exit(-1);
        }

        return y;
    }
}