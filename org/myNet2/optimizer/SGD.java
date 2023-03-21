package org.myNet2.optimizer;

import java.io.PrintWriter;
import java.io.IOException;
import java.util.Random;

import org.myNet2.network.*;
import org.myNet2.lossFunc.*;
import org.myNet2.*;

/**
 * Class for stochastic gradient descent.
 */
public class SGD extends GD {
    /**
     * Constructor fot this class.
     */
    protected SGD(){
        this.rand = new Random(0);
    }

    /**
     * Constructor fot this class.
     * @param net Optimizing network.
     * @param f Loss function of this network.
     */
    public SGD(Network net, LossFunction f){
        this.net = net;
        this.lossFunc = f;
        this.layersLength = net.layers.length;
        this.rand = new Random(0);
    }

    /**
     * Constructor fot this class.
     * @param net Optimizing network.
     * @param f Loss function of this network.
     * @param eta Learning rate.
     */
    public SGD(Network net, LossFunction f, double eta){
        this.net = net;
        this.lossFunc = f;
        this.layersLength = net.layers.length;
        this.eta = eta;
        this.rand = new Random(0);
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
        Matrix y = new Matrix();
        int backNum = x.row % batchSize == 0 ? x.row / batchSize : x.row / batchSize + 1;

        for (int i = 0; i < nEpoch; i++){
            Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
            Matrix[] xs = xt[0];
            Matrix[] ts = xt[1];

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
        Matrix y = new Matrix();
        int backNum = x.row % batchSize == 0 ? x.row / batchSize : x.row / batchSize + 1;
        int valBatchSize = valX.row * batchSize / x.row;

        for (int i = 0; i < nEpoch; i++){
            Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
            Matrix[] xs = xt[0];
            Matrix[] ts = xt[1];
            Matrix[][] valxt = this.makeMiniBatch(valX, valT, valBatchSize, rand);
            Matrix[] valxs = valxt[0];
            Matrix[] valts = valxt[1];
            Matrix valY;

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
        Matrix y = new Matrix();
        int backNum = x.row % batchSize == 0 ? x.row / batchSize : x.row / batchSize + 1;
        double loss = 0.;

        try(
            PrintWriter fp = new PrintWriter(fileName);
        ){
            fp.write("Epoch,loss\n");
            for (int i = 0; i < nEpoch; i++){
                Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
                Matrix[] xs = xt[0];
                Matrix[] ts = xt[1];

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
        Matrix y = new Matrix();
        int backNum = x.row % batchSize == 0 ? x.row / batchSize : x.row / batchSize + 1;
        double loss = 0., valLoss = 0.;

        try(
            PrintWriter fp = new PrintWriter(fileName);
        ){
            fp.write("Epoch,loss,valLoss\n");
            for (int i = 0; i < nEpoch; i++){
                Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
                Matrix[] xs = xt[0];
                Matrix[] ts = xt[1];
                Matrix valY;

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
