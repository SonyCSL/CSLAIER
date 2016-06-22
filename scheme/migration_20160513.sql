alter table Model add column updated_at timestamp;
update Model set updated_at = datetime('now');
alter table Dataset add column category_num int;
alter table Dataset add column file_num int;
