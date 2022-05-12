package org.MyNet2.optimizer;

import org.MyNet2.*;
import org.MyNet2.layer.*;
import org.MyNet2.lossFunc.*;
import org.MyNet2.network.*;

/** 
 * Class for RMSprop.
 */
public class RMSprop extends SGD {
    /** Number for adjust h. */
    protected double alpha = 0.99;
    /** Number for adjust learning rate. */
    protected double h = 10e-8 / this.alpha;
    /** Variable for back propagation. */
    protected double etaDivSqrtH, sum = 0;

    /**
     * Constructor for this class.
     */
    protected RMSprop(){
        ;
    }

    /**
     * Constructor fot this class.
     * @param net Optimizing network.
     * @param f Loss function of this network.
     */
    public RMSprop(Network net, LossFunction f){
        this.net = net;
        this.lossFunc = f;
        this.layersLength = net.layers.length;
    }

    /**
     * Constructor fot this class.
     * @param net Optimizing network.
     * @param f Loss function of this network.
     * @param eta Learning rate.
     * @param h Number for adjust learning rate.
     * @param alpha Number for adjust h.
     */
    public RMSprop(Network net, LossFunction f, double eta, double h, double alpha){
        this.net = net;
        this.lossFunc = f;
        this.layersLength = net.layers.length;
        this.eta = eta;
        this.alpha = alpha;
        this.h = h / this.alpha;
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

        this.h = this.alpha * this.h + (1 - this.alpha) * this.sum;
        this.etaDivSqrtH = this.eta / Math.sqrt(this.h);
        this.sum = 0.;

        Matrix E = this.lossFunc.diff(lastLayer.a, t);
        Matrix f = lastLayer.actFunc.diff(lastLayer.x);
        for (int i = 0; i < lastLayer.nodesNum; i++){
            double num = 0.;
            for (int j = 0; j < x.row; j++){
                num += E.matrix[j][i] * f.matrix[j][i];
            }
            lastLayer.delta.matrix[i][0] = num;
        }

        Matrix gradW = lastLayer.delta.dot(preLayer.a.meanCol().appendCol(1.)).T();
        this.sum += gradW.pow().sum();
        lastLayer.w = lastLayer.w.add(gradW.mult(-this.etaDivSqrtH));
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
            cal = aPre.T().mult(nowLayer.delta.matrix[i][0]);
            this.sum += cal.pow().sum();
            for (int j = 0; j < nowLayer.inNum; j++){
                nowLayer.w.matrix[j][i] -= this.etaDivSqrtH * cal.matrix[j][0];
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

        nowLayer.b = nowLayer.b.add(gradB.mult(-this.etaDivSqrtH));
        nowLayer.w = nowLayer.w.add(gradW.mult(-this.etaDivSqrtH));

        this.sum += gradW.pow().sum();
        this.sum += gradB.pow().sum();

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