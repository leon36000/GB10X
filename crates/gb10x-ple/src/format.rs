#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_rejects_zero_dimensions() {
        assert!(PlePackHeader::new([1; 32], 0, 320, 4096, [2; 32]).is_err());
        assert!(PlePackHeader::new([1; 32], 10, 0, 4096, [2; 32]).is_err());
        assert!(PlePackHeader::new([1; 32], 10, 320, 0, [2; 32]).is_err());
    }

    #[test]
    fn header_keeps_source_and_index_provenance() {
        let header = PlePackHeader::new([0xAA; 32], 100, 320, 4096, [0xBB; 32])
            .expect("valid exact PLEPack header");
        assert_eq!(header.source_digest, [0xAA; 32]);
        assert_eq!(header.index_digest, [0xBB; 32]);
        assert_eq!(header.row_count, 100);
        assert_eq!(header.row_bytes, 320);
        assert_eq!(header.block_bytes, 4096);
    }
}
