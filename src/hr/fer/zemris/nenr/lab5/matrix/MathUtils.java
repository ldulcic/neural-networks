package hr.fer.zemris.apr.lab1;

/**
 * Util for working with <code>double</code> values.
 *
 * Created by luka on 10/21/16.
 */
public class MathUtils {

    private static final double EPSILON = 10E-8;

    /**
     * Checks if two double values are approximately equal with respect to some epsilon value.
     *
     * @return <code>true</code> if <code>value1</code> and <code>value2</code> are equal with respect to epsilon,
     *          <code>false</code> otherwise.
     */
    public static boolean equals(double value1, double value2) {
        return Math.abs(Math.abs(value1) - Math.abs(value2)) < EPSILON;
    }
}
