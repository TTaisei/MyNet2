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
 * Class for gradient descent.
 */
public class GD extends Optimizer {
    /**
     * Constructor fot this class.
     * @param net Optimizing network.
     * @param f Loss function of this network.
     */
    public GD(Network net, LossFunction f){
        this.net = net;
        this.lossFunc = f;
        this.layersLength = net.layers.length;
    }

    /**
     * Constructor fot this class.
     * @param net Optimizing network.
     * @param f Loss function of this network.
     * @param eta Learning rate.
     */
    public GD(Network net, LossFunction f, double eta){
        this.net = net;
        this.lossFunc = f;
        this.layersLength = net.layers.length;
        this.eta = eta;
    }

    /**
     * Doing back propagation.
     * @param x input matrix.
     * @param y Result of forward propagation.
     * @param t Answer.
     */
    // protected void back(Matrix x, Matrix y, Matrix t){
    public void back(Matrix x, Matrix y, Matrix t){
        this.backLastLayer(x, y, t);

        for (int i = this.layersLength-2; i >= 0; i--){
            this.net.layers[i].back(
                this.net.layers[i+1].delta,
                this.net.layers[i+1].w,
                i == 0 ? x : this.net.layers[i-1].a
            );
        }
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch){
        Matrix y = this.net.forward(x);

        for (int i = 0; i < nEpoch; i++){
            System.out.printf("Epoch %d/%d\n", i+1, nEpoch);
            this.back(x, y, t);
            y = this.net.forward(x);
            System.out.printf("loss: %.4f\n", this.lossFunc.calc(y, t));
        }

        return y;
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param valX Input layer for validation.
     * @param valT Answer for validation.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, Matrix valX, Matrix valT){
        Matrix y = this.net.forward(x);
        Matrix valY;

        for (int i = 0; i < nEpoch; i++){
            System.out.printf("Epoch %d/%d\n", i+1, nEpoch);
            this.back(x, y, t);
            valY = this.net.forward(valX);
            y = this.net.forward(x);
            System.out.printf(
                "loss: %.4f - valLoss: %.4f\n",
                this.lossFunc.calc(y, t),
                this.lossFunc.calc(valY, valT)
            );
        }

        return y;
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param fileName Name of logging file.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, String fileName){
        Matrix y = this.net.forward(x);
        double loss = 0.;

        try(
            PrintWriter fp = new PrintWriter(fileName);
        ){
            fp.write("Epoch,loss\n");
            for (int i = 0; i < nEpoch; i++){
                this.back(x, y, t);
                y = this.net.forward(x);
                loss = this.lossFunc.calc(y, t);
                fp.printf("%d,%f\n", i+1, loss);
            }
        }catch (IOException e){
            System.out.println("IO Exception");
            System.exit(-1);
        }

        return y;
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param valX Input layer for validation.
     * @param valT Answer for validation.
     * @param fileName Name of logging file.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, Matrix valX, Matrix valT, String fileName){
        Matrix y = this.net.forward(x);
        Matrix valY;
        double loss = 0., valLoss = 0.;

        try(
            PrintWriter fp = new PrintWriter(fileName);
        ){
            fp.write("Epoch,loss,valLoss\n");
            for (int i = 0; i < nEpoch; i++){
                this.back(x, y, t);
                valY = this.net.forward(valX);
                y = this.net.forward(x);
                loss = this.lossFunc.calc(y, t);
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