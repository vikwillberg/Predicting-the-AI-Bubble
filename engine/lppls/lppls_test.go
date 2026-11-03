package lppls

import (
	"math"
	"testing"
)

// synthetic builds a noiseless LPPLS bubble with a known critical time.
func synthetic(n int, tc, m, w, a, b, c1, c2 float64) ([]float64, []float64) {
	t := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		t[i] = float64(i)
		dt := tc - t[i]
		f := math.Pow(dt, m)
		y[i] = a + b*f + c1*f*math.Cos(w*math.Log(dt)) + c2*f*math.Sin(w*math.Log(dt))
	}
	return t, y
}

func TestSolve4(t *testing.T) {
	a := [4][4]float64{{4, 1, 0, 0}, {1, 3, 1, 0}, {0, 1, 2, 1}, {0, 0, 1, 5}}
	want := [4]float64{1, -2, 3, 0.5}
	var b [4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			b[i] += a[i][j] * want[j]
		}
	}
	got, ok := solve4(a, b)
	if !ok {
		t.Fatal("solver reported singular matrix")
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-10 {
			t.Fatalf("x[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRecoversCriticalTime(t *testing.T) {
	const tcTrue = 520.0
	tt, y := synthetic(500, tcTrue, 0.5, 8.0, 6.0, -0.05, 0.004, -0.003)
	dt := tt[len(tt)-1] - tt[0]
	t2 := tt[len(tt)-1]
	b := Bounds{TcLo: t2 - 0.2*dt, TcHi: t2 + 0.2*dt, MLo: 0.01, MHi: 0.99, WLo: 2, WHi: 25}
	q := Qual{MLo: 0.01, MHi: 0.99, WLo: 2, WHi: 25, TcLoFrac: -0.05, TcHiFrac: 0.1, MinOscillations: 2.5, MinDamping: 0.5}
	fit := FitWindow(tt, y, b, q, DefaultSearch)
	if !fit.OK {
		t.Fatal("fit failed")
	}
	if math.Abs(fit.Tc-tcTrue) > 3 {
		t.Fatalf("tc = %.2f, want %.2f", fit.Tc, tcTrue)
	}
	if fit.B >= 0 || fit.Bubble != 1 {
		t.Fatalf("expected a positive bubble (B<0), got B=%v", fit.B)
	}
}
