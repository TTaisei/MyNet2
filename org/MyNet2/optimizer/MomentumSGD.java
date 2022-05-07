package org.MyNet2.optimizer;

import java.util.ArrayList;

import org.MyNet2.network.*;
import org.MyNet2.lossFunc.*;
import org.MyNet2.layer.*;
import org.MyNet2.*;

public class MomentumSGD extends SGD {
    /** Value of momentum. */
    public double alpha = 0.9;
    /** Amount of change in weight. */
    protected ArrayList<Matrix> dW;
    /** Amount of change in bias. */
    protected ArrayList<Matrix> dB;

    /**
     * Constructor for this class.
     */
    protected MomentumSGD(){
        ;
    }

    /**
     * Constructor fot this class.
     * @param net Optimizing network.
     * @param f Loss function of this network.
     */
    public MomentumSGD(Network net, LossFunction f){
        this.net = net;
        this.lossFunc = f;
        this.layersLength = net.layers.length;
        this.setDW();
    }

    /**
     * Constructor fot this class.
     * @param net Optimizing network.
     * @param f Loss function of this network.
     * @param eta Learning rate.
     * @param alpha 
     */
    public MomentumSGD(Network net, LossFunction f, double eta, double alpha){
        this.net = net;
        this.lossFunc = f;
        this.layersLength = net.layers.length;
        this.eta = eta;
        this.alpha = alpha;
        this.setDW();
    }

    protected void setDW(){
        this.dW = new ArrayList<Matrix>();
        this.dB = new ArrayList<Matrix>();

        for (Layer layer: this.net.layers){
            switch(layer.name){
                case "Dense":
                    this.dW.add(new Matrix(layer.inNum, layer.nodesNum));
                    this.dB.add(new Matrix());
                    break;
                case "Conv":
                    this.dW.add(new Matrix(layer.kernelNum, layer.channelNum*layer.wRow*layer.wCol));
                    this.dB.add(new Matrix(layer.kernelNum, 1));
                    break;
                case "MaxPooling":
                    this.dW.add(new Matrix());
                    this.dB.add(new Matrix());
                    break;
            }
        }
    }

    /**
     * Doing back propagation to last layer.
     * @param x input matrix.
     * @param y Result of forward propagation.
     * @param t Answer.
     */
    @Override
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

        this.dW.set(
            this.layersLength-1,
            this.dW.get(this.layersLength-1).mult(this.alpha).add(
                lastLayer.delta.dot(preLayer.a.meanCol().appendCol(1.)).mult(-this.eta).T()
            )
        );
        lastLayer.w = lastLayer.w.add(this.dW.get(this.layersLength-1));
    }

    /**
     * Doing back propagation.
     * @param num Number of layer.
     * @param aPre Output matrix of previous layer.
     */
    @Override
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
                this.dW.get(num).matrix[j][i] = cal.matrix[j][0] + this.alpha * this.dW.get(num).matrix[j][i];
                nowLayer.w.matrix[j][i] = this.dW.get(num).matrix[j][i];
            }
        }
    }

    /**
     * Doing back propagation.
     * @param num Number of layer.
     * @param aPre Output matrix of previous layer.
     */
    @Override
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

        this.dB.set(
            num,
            this.dB.get(num).mult(this.alpha).add(gradB.mult(-this.eta))
        );
        this.dW.set(
            num,
            this.dW.get(num).mult(this.alpha).add(gradW.mult(-this.eta))
        );

        nowLayer.b = nowLayer.b.add(this.dB.get(num));
        nowLayer.w = nowLayer.w.add(this.dW.get(num));

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
}