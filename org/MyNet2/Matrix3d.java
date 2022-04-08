package org.MyNet2;

import java.util.ArrayList;
import java.util.Random;
import org.MyNet2.*;

/**
 * Class for three dimentional matrix.
 */
public class Matrix3d {
    /** Value of this matrix. */
    public ArrayList<Matrix> matrix;
    /** Shape of this matrix. */
    public int[] shape;

    /**
     * Exit this program.
     * @param msg Message for printing.
     */
    protected void exit(String msg){
        System.out.println(msg);
        System.exit(-1);
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     */
    public Matrix3d(int[] shape){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix>();

        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix(this.shape[1], this.shape[2]));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param num Number to fill.
     */
    public Matrix3d(int[] shape, double num){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix>();

        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix(this.shape[1], this.shape[2], num));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param rand Random instance.
     */
    public Matrix3d(int[] shape, Random rand){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix>();

        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix(this.shape[1], this.shape[2], rand));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param rand Random instance.
     * @param min Number of min for range.
     * @param max Number of max for range.
     */
    public Matrix3d(int[] shape, Random rand, double min, double max){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix>();

        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix(this.shape[1], this.shape[2], rand, min, max));
        }
    }

    @Override
    public Matrix3d clone(){
        Matrix3d rtn = new Matrix3d(this.shape);

        for (int i = 0; i < this.shape[0]; i++){
            rtn.matrix.set(i, this.matrix.get(i).clone());
        }

        return rtn;
    }
}