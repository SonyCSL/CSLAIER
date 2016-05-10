alter table Model add column updated_at timestamp;
update Model set updated_at = datetime('now');