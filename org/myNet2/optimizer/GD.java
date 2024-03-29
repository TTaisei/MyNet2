package org.myNet2.optimizer;

import java.io.PrintWriter;
import java.io.IOException;

import org.myNet2.network.*;
import org.myNet2.lossFunc.*;
import org.myNet2.layer.*;
import org.myNet2.*;

/**
 * Class for gradient descent.
 */
public class GD extends Optimizer {
    /**
     * Constructor for this class.
     */
    protected GD(){
        ;
    }

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
    protected void back(Matrix x, Matrix y, Matrix t){
        this.backLastLayer(x, y, t);

        for (int i = this.layersLength-2; i >= 0; i--){
            switch (this.net.layers[i].name){
                case "Dense":
                    this.backDense(
                        i,
                        i == 0 ? x.meanCol().appendCol(1.) : this.net.layers[i-1].a.meanCol().appendCol(1.)
                    );
                    break;
                case "Conv":
                    this.backConv(
                        i,
                        i == 0 ? x : this.net.layers[i-1].a.meanCol()
                    );
                    break;
                case "MaxPooling":
                    this.backMaxPooling(i);
                    break;
                default:
                    ;
            }
        }
    }

    /**
     * Doing back propagation to last layer.
     * @param x input matrix.
     * @param y Result of forward propagation.
     * @param t Answer.
     */
    protected void backLastLayer(Matrix x, Matrix y, Matrix t){
        Layer lastLayer = this.net.layers[this.layersLength-1];
        Layer preLayer = this.net.layers[this.layersLength-2];

        Matrix E = this.lossFunc.diff(lastLayer.a, t);
        Matrix f = lastLayer.actFunc.diff(lastLayer.x);
        for (int i = 0; i < lastLayer.nodesNum; i++){
            double num = 0.;
            for (int j = 0; j < x.row; j++){
                num += E.matrix[j][i] * f.matrix[j][i];
            }
            lastLayer.delta.matrix[i][0] = num;
        }

        lastLayer.w = lastLayer.w.add(lastLayer.delta.dot(preLayer.a.meanCol().appendCol(1.)).mult(-this.eta).T());
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
            cal = nowLayer.w.getCol(i).add(aPre.T().mult(-this.eta*nowLayer.delta.matrix[i][0]));
            for (int j = 0; j < nowLayer.inNum; j++){
                nowLayer.w.matrix[j][i] = cal.matrix[j][0];
            }
        }
    }

    /**
     * Doing back propagation.
     * @param num Number of layer.
     * @param aPre Output matrix of previous layer.
     */
    protected void backConv(int num, Matrix aPre){
        Matrix deltaNext = this.net.layers[num+1].delta;
        Layer nowLayer = this.net.layers[num];
        Matrix xMeanCol = nowLayer.x.meanCol();
        int cMult = nowLayer.wRow * nowLayer.wCol;
        int kMult = nowLayer.outRow * nowLayer.outCol;
        int iMult = nowLayer.outCol;

        Matrix gradW = new Matrix(nowLayer.w.row, nowLayer.w.col);
        Matrix gradB = new Matrix(nowLayer.b.row, nowLayer.b.col);

        for (int k = 0; k < nowLayer.kernelNum; k++){
            double d = 0.;
            Matrix fD = nowLayer.actFunc.diff(xMeanCol.add(nowLayer.b.matrix[k][0]));

            for (int i = 0; i < nowLayer.outRow; i++){
                for (int j = 0; j < nowLayer.outCol; j++){
                    d = deltaNext.matrix[k][iMult*i + j] * fD.matrix[0][kMult*k + iMult*i + j];
                    gradB.matrix[k][0] += d;

                    for (int c = 0; c < nowLayer.channelNum; c++){
                        for (int p = 0; p < nowLayer.wRow; p++){
                            for (int q = 0; q < nowLayer.wCol; q++){
                                gradW.matrix[k][cMult*c + nowLayer.wCol*p + q] += d * aPre.matrix[c][nowLayer.inCol*(i+p) + j+q];
                            }
                        }
                    }
                }
            }
        }

        nowLayer.b = nowLayer.b.add(gradB.mult(-this.eta));
        nowLayer.w = nowLayer.w.add(gradW.mult(-this.eta));

        double d;
        for (int k = 0; k < nowLayer.kernelNum; k++){
            Matrix fD = nowLayer.actFunc.diff(xMeanCol.add(nowLayer.b.matrix[k][0]));
            for (int i = 0; i < nowLayer.inRow; i++){
                for (int j = 0; j < nowLayer.inCol; j++){
                    for (int c = 0; c < nowLayer.channelNum; c++){
                        for (int p = 0; p < nowLayer.wRow; p++){
                            for (int q = 0; q < nowLayer.wCol; q++){
                                if ((i - (nowLayer.wRow-1) - p < 0) || (j - (nowLayer.wCol - 1) - q < 0)){
                                    d = 0.;
                                }else{
                                    d = deltaNext.matrix[k][iMult*(i-(nowLayer.wRow-1)-p) + j-(nowLayer.wCol-1)-q]
                                        * fD.matrix[0][kMult*k + iMult*(i-(nowLayer.wRow-1)-p) + j-(nowLayer.wCol-1)-q]
                                        * nowLayer.w.matrix[k][cMult*c + nowLayer.wCol*p + q];
                                }
                                nowLayer.delta.matrix[c][iMult*i + j] += d;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Doing back propagation.
     * @param num Number of layer.
     */
    protected void backMaxPooling(int num){
        Matrix aPre = this.net.layers[num-1].a.meanCol();
        Matrix deltaNext = this.net.layers[num+1].delta;
        Layer nowLayer = this.net.layers[num];
        int kMult = nowLayer.outRow * nowLayer.outCol;
        int iMult = nowLayer.outCol;
        int poolSize = nowLayer.poolSize;

        for (int k = 0; k < nowLayer.kernelNum; k++){
            for (int i = 0; i < nowLayer.outRow; i++){
                for (int j = 0; j < nowLayer.outCol; j++){
                    for (int p = 0; p < poolSize; p++){
                        for (int q = 0; q < poolSize; q++){
                            if (nowLayer.a.matrix[0][kMult*k + iMult*i + j]
                            == aPre.matrix[0][kMult*k + iMult*(poolSize*i+p) + poolSize*j+q]){
                                nowLayer.delta.matrix[k][iMult*(poolSize*i+p) + poolSize*j+q] = deltaNext.matrix[k][iMult*i + j];
                            }else{
                                nowLayer.delta.matrix[k][iMult*(poolSize*i+p) + poolSize*j+q] = 0.;
                            }
                        }
                    }
                }
            }
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
