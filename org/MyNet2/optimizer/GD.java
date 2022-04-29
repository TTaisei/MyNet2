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
            switch (this.net.layers[i].name){
                case "Dense":
                    this.backDense(
                        i,
                        i == 0 ? x.meanCol().appendCol(1.) : this.net.layers[i-1].a.meanCol().appendCol(1.)
                    );
                case "Conv":
                    this.backConv(i);
                case "MaxPooling":
                    this.backMaxPooling(i);
                default:
                    ;
            }
        }
    }

    /**
     * Doing back propagation.
     * @param num Number of layer.
     * @param aPre Output matrix of previous layer.
     */
    protected void backDense(int num, Matrix aPre){
        Matrix deltaNext = this.net.layers[num+1].delta;
        Matrix wNext = this.net.layers[num+1].w;
        Layer nowLayer = this.net.layers[num];

        double deltaEle = 0.;
        Matrix cal;

        for (int i = 0; i < deltaNext.row; i++){
            deltaEle += wNext.getCol(i).mult(deltaNext.matrix[i][0]).sum();
        }
        for (int i = 0; i < nowLayer.nodesNum; i++){
            nowLayer.delta.matrix[i][0] = 
                deltaEle * nowLayer.actFunc.diff(nowLayer.x.getCol(i)).meanCol().matrix[0][0];
            cal = nowLayer.w.getCol(i).add(aPre.T().mult(-this.eta));
            for (int j = 0; j < nowLayer.inNum; j++){
                nowLayer.w.matrix[j][i] = cal.matrix[j][0];
            }
        }
    }

    /**
     * Doing back propagation.
     * @param num Number of layer.
     */
    protected void backConv(int num){
        ;
    }

    /**
     * Doing back propagation.
     * @param num Number of layer.
     */
    protected void backMaxPooling(int num){
        ;
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