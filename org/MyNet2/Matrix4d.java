package org.MyNet2;

import java.util.ArrayList;
import java.util.Random;

/**
 * Class for four dimentional matrix.
 */
public class Matrix4d {
    /** Value of this matrix. */
    public ArrayList<Matrix3d> matrix;
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
    public Matrix4d(int[] shape){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix3d>();

        int[] shapeElement = {this.shape[1], this.shape[2], this.shape[3]};
        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix3d(shapeElement));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param num Number to fill.
     */
    public Matrix4d(int[] shape, double num){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix3d>();

        int[] shapeElement = {this.shape[1], this.shape[2], this.shape[3]};
        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix3d(shapeElement, num));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param rand Random instance.
     */
    public Matrix4d(int[] shape, Random rand){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix3d>();

        int[] shapeElement = {this.shape[1], this.shape[2], this.shape[3]};
        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix3d(shapeElement, rand));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param rand Random instance.
     * @param min Number of min for range.
     * @param max Number of max for range.
     */
    public Matrix4d(int[] shape, Random rand, double min, double max){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix3d>();

        int[] shapeElement = {this.shape[1], this.shape[2], this.shape[3]};
        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix3d(shapeElement, rand, min, max));
        }
    }

    @Override
    public Matrix4d clone(){
        Matrix4d rtn = new Matrix4d(this.shape);

        for (int i = 0; i < this.shape[0]; i++){
            rtn.matrix.set(i, this.matrix.get(i).clone());
        }

        return rtn;
    }
}