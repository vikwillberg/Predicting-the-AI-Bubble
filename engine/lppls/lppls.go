// Package lppls fits the Log-Periodic Power Law Singularity model of
// Johansen, Ledoit & Sornette (2000) using the Filimonov & Sornette (2013)
// reformulation, in which four of the seven parameters are solved exactly
// by least squares and only (tc, m, omega) are searched non-linearly:
//
//	ln p(t) = A + B f + C1 f cos(omega ln|tc-t|) + C2 f sin(omega ln|tc-t|),  f = |tc-t|^m
//
// The search is deterministic (coarse grid, then Nelder-Mead from the best
// grid points), so this implementation and the Python reference in
// icarus/models/lppls.py return the same fits. tests/test_lppls.py checks that.
package lppls

import (
	"math"
	"sort"
)

// Bounds are the search box for the non-linear parameters.
type Bounds struct {
	TcLo, TcHi float64
	MLo, MHi   float64
	WLo, WHi   float64
}

// Qual holds the qualification filters for a "valid" bubble fit.
type Qual struct {
	MLo, MHi        float64
	WLo, WHi        float64
	TcLoFrac        float64 // tc >= t2 + TcLoFrac*(t2-t1)  (negative = before t2)
	TcHiFrac        float64 // tc <= t2 + TcHiFrac*(t2-t1)
	MinOscillations float64
	MinDamping      float64
}

// Search controls the grid resolution and the local optimiser.
type Search struct {
	GridTc, GridM, GridW int
	Starts               int
	MaxIter              int
}

// DefaultSearch is shared with the Python reference implementation.
var DefaultSearch = Search{GridTc: 12, GridM: 8, GridW: 12, Starts: 3, MaxIter: 400}

// Fit is one calibrated window.
type Fit struct {
	T1           float64 `json:"t1"`
	T2           float64 `json:"t2"`
	Tc           float64 `json:"tc"`
	M            float64 `json:"m"`
	Omega        float64 `json:"omega"`
	A            float64 `json:"A"`
	B            float64 `json:"B"`
	C1           float64 `json:"C1"`
	C2           float64 `json:"C2"`
	SSE          float64 `json:"sse"`
	Oscillations float64 `json:"oscillations"`
	Damping      float64 `json:"damping"`
	Qualified    bool    `json:"qualified"`
	Bubble       int     `json:"bubble"` // +1 positive bubble (B<0), -1 negative bubble (B>0)
	OK           bool    `json:"ok"`
}

// workspace avoids allocations inside the hot loop.
type workspace struct {
	x [4][]float64
}

func newWorkspace(n int) *workspace {
	ws := &workspace{}
	for k := range ws.x {
		ws.x[k] = make([]float64, n)
	}
	return ws
}

// solve4 solves a 4x4 linear system by Gaussian elimination with partial pivoting.
func solve4(a [4][4]float64, b [4]float64) ([4]float64, bool) {
	var x [4]float64
	scale := 0.0
	for i := 0; i < 4; i++ {
		scale = math.Max(scale, math.Abs(a[i][i]))
	}
	if scale == 0 {
		return x, false
	}
	for col := 0; col < 4; col++ {
		piv := col
		for r := col + 1; r < 4; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[piv][col]) {
				piv = r
			}
		}
		if math.Abs(a[piv][col]) < 1e-13*scale {
			return x, false
		}
		a[col], a[piv] = a[piv], a[col]
		b[col], b[piv] = b[piv], b[col]
		for r := col + 1; r < 4; r++ {
			f := a[r][col] / a[col][col]
			for c := col; c < 4; c++ {
				a[r][c] -= f * a[col][c]
			}
			b[r] -= f * b[col]
		}
	}
	for r := 3; r >= 0; r-- {
		s := b[r]
		for c := r + 1; c < 4; c++ {
			s -= a[r][c] * x[c]
		}
		x[r] = s / a[r][r]
	}
	return x, true
}

// Linear solves (A, B, C1, C2) for fixed (tc, m, omega) and returns the SSE.
func Linear(t, y []float64, tc, m, w float64, ws *workspace) ([4]float64, float64, bool) {
	n := len(t)
	if ws == nil {
		ws = newWorkspace(n)
	}
	var xtx [4][4]float64
	var xty [4]float64
	for i := 0; i < n; i++ {
		dt := math.Abs(tc - t[i])
		if dt < 1e-8 {
			dt = 1e-8
		}
		lg := math.Log(dt)
		f := math.Exp(m * lg)
		s, c := math.Sincos(w * lg)
		x0, x1, x2, x3 := 1.0, f, f*c, f*s
		ws.x[0][i], ws.x[1][i], ws.x[2][i], ws.x[3][i] = x0, x1, x2, x3
		xs := [4]float64{x0, x1, x2, x3}
		for a := 0; a < 4; a++ {
			xty[a] += xs[a] * y[i]
			for b := a; b < 4; b++ {
				xtx[a][b] += xs[a] * xs[b]
			}
		}
	}
	for a := 0; a < 4; a++ {
		for b := 0; b < a; b++ {
			xtx[a][b] = xtx[b][a]
		}
	}
	beta, ok := solve4(xtx, xty)
	if !ok {
		return beta, math.Inf(1), false
	}
	sse := 0.0
	for i := 0; i < n; i++ {
		r := y[i] - (beta[0]*ws.x[0][i] + beta[1]*ws.x[1][i] + beta[2]*ws.x[2][i] + beta[3]*ws.x[3][i])
		sse += r * r
	}
	if math.IsNaN(sse) {
		return beta, math.Inf(1), false
	}
	return beta, sse, true
}

func linspace(lo, hi float64, k int) []float64 {
	if k == 1 {
		return []float64{(lo + hi) / 2}
	}
	out := make([]float64, k)
	for i := range out {
		out[i] = lo + (hi-lo)*float64(i)/float64(k-1)
	}
	return out
}

func clamp01(v float64) float64 { return math.Min(1, math.Max(0, v)) }

type objective func(u [3]float64) float64

// nelderMead minimises f over the unit cube (points are clamped into it).
// Standard coefficients: reflection 1, expansion 2, contraction 0.5, shrink 0.5.
func nelderMead(f objective, x0 [3]float64, maxIter int) ([3]float64, float64) {
	const nd = 3
	var simplex [nd + 1][3]float64
	var fv [nd + 1]float64
	for k := 0; k < nd; k++ {
		x0[k] = clamp01(x0[k])
	}
	simplex[0] = x0
	for k := 0; k < nd; k++ {
		p := x0
		step := 0.1
		if p[k]+step > 1 {
			step = -step
		}
		p[k] += step
		simplex[k+1] = p
	}
	for i := range simplex {
		fv[i] = f(simplex[i])
	}
	clampPt := func(p [3]float64) [3]float64 {
		for k := 0; k < nd; k++ {
			p[k] = clamp01(p[k])
		}
		return p
	}
	for iter := 0; iter < maxIter; iter++ {
		// order (stable sort by value, ties keep index order)
		idx := []int{0, 1, 2, 3}
		sort.SliceStable(idx, func(a, b int) bool { return fv[idx[a]] < fv[idx[b]] })
		var s2 [nd + 1][3]float64
		var f2 [nd + 1]float64
		for i, j := range idx {
			s2[i], f2[i] = simplex[j], fv[j]
		}
		simplex, fv = s2, f2

		spread := math.Abs(fv[nd] - fv[0])
		size := 0.0
		for i := 1; i <= nd; i++ {
			for k := 0; k < nd; k++ {
				size = math.Max(size, math.Abs(simplex[i][k]-simplex[0][k]))
			}
		}
		if spread <= 1e-12*(1+math.Abs(fv[0])) && size < 1e-7 {
			break
		}
		var c [3]float64
		for i := 0; i < nd; i++ {
			for k := 0; k < nd; k++ {
				c[k] += simplex[i][k] / nd
			}
		}
		var xr [3]float64
		for k := 0; k < nd; k++ {
			xr[k] = c[k] + (c[k] - simplex[nd][k])
		}
		xr = clampPt(xr)
		fr := f(xr)
		switch {
		case fr < fv[0]:
			var xe [3]float64
			for k := 0; k < nd; k++ {
				xe[k] = c[k] + 2*(xr[k]-c[k])
			}
			xe = clampPt(xe)
			fe := f(xe)
			if fe < fr {
				simplex[nd], fv[nd] = xe, fe
			} else {
				simplex[nd], fv[nd] = xr, fr
			}
		case fr < fv[nd-1]:
			simplex[nd], fv[nd] = xr, fr
		default:
			var xc [3]float64
			if fr < fv[nd] {
				for k := 0; k < nd; k++ {
					xc[k] = c[k] + 0.5*(xr[k]-c[k])
				}
			} else {
				for k := 0; k < nd; k++ {
					xc[k] = c[k] + 0.5*(simplex[nd][k]-c[k])
				}
			}
			xc = clampPt(xc)
			fc := f(xc)
			if fc < math.Min(fr, fv[nd]) {
				simplex[nd], fv[nd] = xc, fc
			} else {
				for i := 1; i <= nd; i++ {
					for k := 0; k < nd; k++ {
						simplex[i][k] = simplex[0][k] + 0.5*(simplex[i][k]-simplex[0][k])
					}
					fv[i] = f(simplex[i])
				}
			}
		}
	}
	best := 0
	for i := 1; i <= nd; i++ {
		if fv[i] < fv[best] {
			best = i
		}
	}
	return simplex[best], fv[best]
}

// FitWindow calibrates the LPPLS model on one window (t and y share length).
func FitWindow(t, y []float64, b Bounds, q Qual, s Search) Fit {
	n := len(t)
	ws := newWorkspace(n)
	t1, t2 := t[0], t[n-1]
	toParams := func(u [3]float64) (float64, float64, float64) {
		return b.TcLo + clamp01(u[0])*(b.TcHi-b.TcLo),
			b.MLo + clamp01(u[1])*(b.MHi-b.MLo),
			b.WLo + clamp01(u[2])*(b.WHi-b.WLo)
	}
	obj := func(u [3]float64) float64 {
		tc, m, w := toParams(u)
		_, sse, ok := Linear(t, y, tc, m, w, ws)
		if !ok {
			return math.Inf(1)
		}
		return sse
	}

	type cand struct {
		u   [3]float64
		sse float64
	}
	var cands []cand
	gT, gM, gW := linspace(0, 1, s.GridTc), linspace(0, 1, s.GridM), linspace(0, 1, s.GridW)
	for _, a := range gT {
		for _, bm := range gM {
			for _, c := range gW {
				u := [3]float64{a, bm, c}
				cands = append(cands, cand{u, obj(u)})
			}
		}
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].sse < cands[j].sse })

	bestU, bestF := cands[0].u, cands[0].sse
	starts := s.Starts
	if starts > len(cands) {
		starts = len(cands)
	}
	for k := 0; k < starts; k++ {
		u, fval := nelderMead(obj, cands[k].u, s.MaxIter)
		if fval < bestF {
			bestU, bestF = u, fval
		}
	}
	tc, m, w := toParams(bestU)
	beta, sse, ok := Linear(t, y, tc, m, w, ws)
	fit := Fit{T1: t1, T2: t2, Tc: tc, M: m, Omega: w, A: beta[0], B: beta[1], C1: beta[2], C2: beta[3], SSE: sse, OK: ok}
	if !ok {
		return fit
	}
	Qualify(&fit, q)
	return fit
}

// Qualify applies the Sornette et al. (2015) filters.
func Qualify(f *Fit, q Qual) {
	dt := f.T2 - f.T1
	c := math.Hypot(f.C1, f.C2)
	num := math.Max(math.Abs(f.Tc-f.T1), 1e-8)
	den := math.Max(math.Abs(f.Tc-f.T2), 1e-8)
	f.Oscillations = (f.Omega / math.Pi) * math.Log(num/den)
	if c > 0 {
		f.Damping = (f.M * math.Abs(f.B)) / (f.Omega * c)
	} else {
		f.Damping = math.Inf(1)
	}
	if f.B < 0 {
		f.Bubble = 1
	} else {
		f.Bubble = -1
	}
	f.Qualified = f.M >= q.MLo && f.M <= q.MHi &&
		f.Omega >= q.WLo && f.Omega <= q.WHi &&
		f.Tc >= f.T2+q.TcLoFrac*dt && f.Tc <= f.T2+q.TcHiFrac*dt &&
		f.Oscillations >= q.MinOscillations &&
		f.Damping >= q.MinDamping
}
