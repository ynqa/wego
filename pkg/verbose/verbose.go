package verbose

type Verbose struct {
	flag bool
}

func New(flag bool) *Verbose {
	return &Verbose{
		flag: flag,
	}
}

func (v *Verbose) Do(fn func()) {
	if v.flag {
		fn()
	}
}
