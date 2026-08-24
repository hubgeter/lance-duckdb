use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use lance_core::datatypes::{format_field_path, Schema as LanceSchema};

use crate::constants::{DISTANCE_COLUMN, HYBRID_SCORE_COLUMN, SCORE_COLUMN};

fn format_top_level_field_path(name: &str) -> String {
    format_field_path(&[name])
}

pub(crate) fn format_projection_columns<'a>(
    columns: impl IntoIterator<Item = &'a str>,
    schema: &LanceSchema,
) -> Vec<String> {
    columns
        .into_iter()
        .map(|column| {
            if schema.fields.iter().any(|field| field.name == column) {
                format_top_level_field_path(column)
            } else {
                // Nested paths and virtual columns already use Lance field-path
                // syntax and must not be quoted as one top-level name.
                column.to_string()
            }
        })
        .collect()
}

pub(crate) fn build_base_projection(schema: &Schema) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        cols.push(format_top_level_field_path(field.name()));
    }
    cols.into()
}

pub(crate) fn build_fts_projection(base_projection: &Arc<[String]>) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(base_projection.len() + 1);
    cols.extend(base_projection.iter().cloned());
    cols.push(SCORE_COLUMN.to_string());
    cols.into()
}

pub(crate) fn build_knn_projection(base_projection: &Arc<[String]>) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(base_projection.len() + 1);
    cols.extend(base_projection.iter().cloned());
    cols.push(DISTANCE_COLUMN.to_string());
    cols.into()
}

pub(crate) fn build_hybrid_schema(schema: &Schema) -> Arc<Schema> {
    let mut fields = Vec::with_capacity(schema.fields().len() + 3);
    for field in schema.fields() {
        fields.push(field.clone());
    }
    fields.push(Arc::new(Field::new(
        DISTANCE_COLUMN,
        DataType::Float32,
        true,
    )));
    fields.push(Arc::new(Field::new(SCORE_COLUMN, DataType::Float32, true)));
    fields.push(Arc::new(Field::new(
        HYBRID_SCORE_COLUMN,
        DataType::Float32,
        true,
    )));
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}
