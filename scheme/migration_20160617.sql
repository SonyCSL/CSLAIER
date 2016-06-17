alter table Model add column framework text;
update Model set framework = 'chainer';