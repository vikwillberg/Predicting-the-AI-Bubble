// Command lppls is the high-throughput LPPLS scanner used by ICARUS.
//
// It reads one JSON request on stdin (a log-price series plus a list of
// windows), fits every window in parallel on all CPU cores, and writes the
// fits as JSON to stdout. Scanning tens of thousands of windows (the
// multi-scale LPPLS confidence indicator) takes minutes here, against hours
// for a single-threaded Python loop.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	"github.com/vikwillberg/predicting-the-ai-bubble/engine/lppls"
)

type job struct {
	Start int `json:"start"` // inclusive index into t/y
	End   int `json:"end"`   // inclusive index into t/y
}

type request struct {
	T         []float64  `json:"t"`
	Y         []float64  `json:"y"`
	Jobs      []job      `json:"jobs"`
	TcFitFrac float64    `json:"tc_fit_frac"`
	MRange    [2]float64 `json:"m_range"`
	WRange    [2]float64 `json:"omega_range"`
	Qual      struct {
		TcLoFrac        float64 `json:"tc_lo_frac"`
		TcHiFrac        float64 `json:"tc_hi_frac"`
		MinOscillations float64 `json:"min_oscillations"`
		MinDamping      float64 `json:"min_damping"`
	} `json:"qual"`
	Search  *lppls.Search `json:"search"`
	Workers int           `json:"workers"`
}

type result struct {
	Job int `json:"job"`
	lppls.Fit
}

func main() {
	var req request
	dec := json.NewDecoder(bufio.NewReaderSize(os.Stdin, 1<<20))
	if err := dec.Decode(&req); err != nil {
		fmt.Fprintln(os.Stderr, "lppls: bad request:", err)
		os.Exit(2)
	}
	if len(req.T) != len(req.Y) {
		fmt.Fprintln(os.Stderr, "lppls: t and y lengths differ")
		os.Exit(2)
	}
	search := lppls.DefaultSearch
	if req.Search != nil {
		search = *req.Search
	}
	workers := req.Workers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	qual := lppls.Qual{
		MLo: req.MRange[0], MHi: req.MRange[1], WLo: req.WRange[0], WHi: req.WRange[1],
		TcLoFrac: req.Qual.TcLoFrac, TcHiFrac: req.Qual.TcHiFrac,
		MinOscillations: req.Qual.MinOscillations, MinDamping: req.Qual.MinDamping,
	}

	results := make([]result, len(req.Jobs))
	next := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				j := req.Jobs[i]
				if j.Start < 0 || j.End >= len(req.T) || j.End-j.Start < 10 {
					results[i] = result{Job: i}
					continue
				}
				t := req.T[j.Start : j.End+1]
				y := req.Y[j.Start : j.End+1]
				dt := t[len(t)-1] - t[0]
				t2 := t[len(t)-1]
				b := lppls.Bounds{
					TcLo: t2 - req.TcFitFrac*dt, TcHi: t2 + req.TcFitFrac*dt,
					MLo: req.MRange[0], MHi: req.MRange[1],
					WLo: req.WRange[0], WHi: req.WRange[1],
				}
				results[i] = result{Job: i, Fit: lppls.FitWindow(t, y, b, qual, search)}
			}
		}()
	}
	for i := range req.Jobs {
		next <- i
	}
	close(next)
	wg.Wait()

	// encoding/json rejects Inf and NaN, so map them to sentinel values.
	for i := range results {
		f := &results[i].Fit
		for _, p := range []*float64{&f.SSE, &f.Damping, &f.Oscillations, &f.Tc, &f.M, &f.Omega,
			&f.A, &f.B, &f.C1, &f.C2} {
			if math.IsNaN(*p) {
				*p = 0
				f.OK, f.Qualified = false, false
			} else if math.IsInf(*p, 1) {
				*p = math.MaxFloat64
			} else if math.IsInf(*p, -1) {
				*p = -math.MaxFloat64
			}
		}
	}

	out := bufio.NewWriterSize(os.Stdout, 1<<20)
	defer out.Flush()
	if err := json.NewEncoder(out).Encode(results); err != nil {
		fmt.Fprintln(os.Stderr, "lppls: encode:", err)
		os.Exit(1)
	}
}
