package encode

// data structure for co-occurrence mapping:
// - https://blog.chewxy.com/2017/07/12/21-bits-english/

// EncodeBigram creates id between two words.
func EncodeBigram(l1, l2 uint64) uint64 {
	if l1 < l2 {
		return encode(l1, l2)
	} else {
		return encode(l2, l1)
	}
}

func encode(l1, l2 uint64) uint64 {
	return l1 | (l2 << 32)
}

// DecodeBigram reverts pair id to two word ids.
func DecodeBigram(pid uint64) (uint64, uint64) {
	f := pid >> 32
	return pid - (f << 32), f
}
