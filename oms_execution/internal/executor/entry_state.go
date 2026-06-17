package executor

import (
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func (s *Service) pendingEntryIsCancelling(p *models.PendingEntry) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	cur, ok := s.pending[p.Symbol]
	return ok && cur == p && cur.State == models.PendingEntryStateCancelling
}

// beginPendingEntryCancel marks a pending entry as cancelling and blocks overlapping place/reprice.
func (s *Service) beginPendingEntryCancel(p *models.PendingEntry) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	cur, ok := s.pending[p.Symbol]
	if !ok || cur != p {
		return false
	}
	if cur.State == models.PendingEntryStateCancelling {
		return false
	}
	cur.State = models.PendingEntryStateCancelling
	p.State = models.PendingEntryStateCancelling
	return true
}

func (s *Service) releasePendingEntryCancelIfStuck(p *models.PendingEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cur, ok := s.pending[p.Symbol]
	if !ok || cur != p {
		return
	}
	if cur.State == models.PendingEntryStateCancelling {
		cur.State = models.PendingEntryStateActive
		p.State = models.PendingEntryStateActive
	}
}

func (s *Service) setPendingEntryActive(p *models.PendingEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cur, ok := s.pending[p.Symbol]
	if !ok || cur != p {
		return
	}
	cur.State = models.PendingEntryStateActive
	p.State = models.PendingEntryStateActive
}
