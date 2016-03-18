create table if not exists Model(
    id integer primary key AUTOINCREMENT,
    name text unique not null,
    epoch integer Default 1,
    algorithm text,
    network_name text,
    is_trained integer check(is_trained = 0 or is_trained = 1 or is_trained = 2) Default 0,
    network_path text,
    trained_model_path text,
    graph_data_path text,
    line_graph_data_path text,
    dataset_id integer,
    prepared_file_path integer,
    created_at timestamp default current_timestamp,
    pid integer Default null,
    resize_mode text,
    channels integer Default 3,
    type text
);

create table if not exists Dataset(
    id integer primary key AUTOINCREMENT,
    name text unique not null,
    dataset_path text,
    updated_at timestamp,
    created_at timestamp default current_timestamp,
    type text
);
