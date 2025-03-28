from bech32 import bech32_decode, convertbits

def sui_privkey_to_hex(privkey_bech32: str) -> str:
    hrp, data = bech32_decode(privkey_bech32)
    if hrp != "suiprivkey":
        raise ValueError(f"Prefix(HRP)가 'suiprivkey'가 아닙니다. 실제: {hrp}")

    # 5비트 -> 8비트 변환 (리턴 값은 list[int] 형태)
    converted = convertbits(data, 5, 8, False)
    if converted is None:
        raise ValueError("Bech32 -> bytes 변환에 실패했습니다.")

    # 첫 1바이트(스킴) 제외, 나머지 바이트가 실제 키 데이터
    scheme_byte = converted[0]
    key_list = converted[1:]  # list of int

    # list[int] -> bytes 변환
    key_bytes = bytes(key_list)

    # 바이트 배열을 16진수 문자열로 변환
    hex_key = key_bytes.hex()

    return f"0x{hex_key}"

if __name__ == "__main__":
    sui_privkey_str = "suiprivkey1...yourkey"
    hex_privkey = sui_privkey_to_hex(sui_privkey_str)

    print("원본  Bech32 privkey:", sui_privkey_str)
    print("변환된 0x Hex string:", hex_privkey)
